"""
Two-Pass Spectral Clustering for Speaker Diarization (v2)

Research-backed improvements over v1:
  1. Affinity refinement pipeline (CropDiag → GaussianBlur → RowWiseThreshold → Symmetrize → Diffuse)
  2. Embedding whitening for same-gender speaker separation
  3. Conversational temporal constraints (adjacent = different speaker, skip-1 = same)
  4. p-neighbor affinity pruning (NME-SC / SC-pNA approach)
  5. NME eigengap auto speaker count estimation
  6. Quality-aware Pass 2 weighting (short segments → more temporal, less embedding)
  7. Balanced anchor sampling via farthest-first traversal
  8. Iterative centroid refinement after Pass 2
  9. assign_labels="discretize" for deterministic spectral decomposition

References:
  - Park et al. (2020) "Auto-Tuning Spectral Clustering for SV" (NME-SC)
  - Wang et al. (2018) "Speaker Diarization with LSTM" (affinity refinement)
  - Raghav et al. (2025) "SC-pNA" (adaptive per-row pruning)
  - Turn-to-Diarize (Google 2021) — conversational constraints
"""

import logging
import numpy as np
from typing import Optional, Sequence
from collections import Counter

logger = logging.getLogger("nexus.voice.spectral")

# ── Configuration ──
ANCHOR_MIN_DURATION_MS = 1500
ANCHOR_FALLBACK_MS = 800
MIN_ANCHORS = 4
TEMPORAL_WEIGHT = 0.10          # Moderate temporal boost (v1 was 0.12, v2 was 0.05)
TEMPORAL_WINDOW_SEC = 25.0      # Moderate window (v1 was 30, v2 was 10 — too narrow)
GAUSSIAN_BLUR_SIGMA = 1.0       # For affinity refinement
P_NEIGHBOR_RATIO = 0.30         # Keep top 30% neighbors per row (10% was too aggressive)
CENTROID_REFINEMENT_ITERS = 3   # Post-clustering centroid refinement passes
CENTROID_TOP_K_RATIO = 0.6      # Use top 60% most similar members for centroid


# ═══════════════════════════════════════════════════════════════
# AFFINITY REFINEMENT PIPELINE (Wang et al. 2018 / Google)
# ═══════════════════════════════════════════════════════════════

def _crop_diagonal(affinity: np.ndarray) -> np.ndarray:
    """Replace diagonal with max non-diagonal value per row."""
    A = affinity.copy()
    # Set diagonal to 1.0 (self-similarity)
    np.fill_diagonal(A, 1.0)
    return A


def _gaussian_blur(affinity: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth affinity matrix with Gaussian kernel to denoise."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(affinity, sigma=sigma)


def _row_wise_threshold(affinity: np.ndarray, p_percentile: float = 0.95) -> np.ndarray:
    """For each row, zero out values below the p-th percentile."""
    A = affinity.copy()
    n = A.shape[0]
    for i in range(n):
        threshold = np.percentile(A[i], p_percentile * 100)
        A[i, A[i] < threshold] = 0
    return A


def _symmetrize(affinity: np.ndarray) -> np.ndarray:
    """Restore symmetry: A = max(A, A^T)."""
    return np.maximum(affinity, affinity.T)


def _p_neighbor_prune(affinity: np.ndarray, p_neighbors: int) -> np.ndarray:
    """Keep only top-p similarities per row, zero the rest (NME-SC)."""
    pruned = np.zeros_like(affinity)
    n = affinity.shape[0]
    p = max(2, min(p_neighbors, n - 1))
    for i in range(n):
        top_p = np.argsort(affinity[i])[-p:]
        pruned[i, top_p] = affinity[i, top_p]
    return _symmetrize(pruned)


def refine_affinity(raw_cosine_sim: np.ndarray, n_anchors: int) -> np.ndarray:
    """
    Affinity refinement pipeline (Wang et al. 2018, simplified).
    CropDiagonal → GaussianBlur → RowWiseThreshold → Symmetrize → pNeighborPrune

    NOTE: Diffusion (A @ A.T) removed — it creates second-order connections that
    merge speakers sharing common neighbors (caused Julian+Frank merge).
    """
    A = _crop_diagonal(raw_cosine_sim)
    A = _gaussian_blur(A, sigma=GAUSSIAN_BLUR_SIGMA)
    A = _row_wise_threshold(A, p_percentile=0.90)  # Relaxed from 0.95 (keep top 10% not 5%)
    A = _symmetrize(A)
    # Normalize to [0, 1]
    if A.max() > 0:
        A = A / A.max()
    # p-neighbor pruning (30% of anchors, min 3)
    p = max(3, int(n_anchors * P_NEIGHBOR_RATIO))
    A = _p_neighbor_prune(A, p)
    # Final symmetrize + set diagonal to 1.0
    A = _symmetrize(A)
    np.fill_diagonal(A, 1.0)
    return A


# ═══════════════════════════════════════════════════════════════
# EMBEDDING PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def whiten_embeddings(embeddings: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Partial whitening: alpha-blend between original and fully whitened embeddings.

    Full whitening (alpha=1.0) improves same-gender separation but can collapse
    different speakers who share variance directions (e.g. two males with similar
    pitch range). Partial whitening (alpha=0.3) gives a mild spread without
    destroying the original embedding geometry.
    """
    if alpha <= 0:
        return embeddings.copy()

    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = np.cov(centered.T)
    cov += 1e-6 * np.eye(cov.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))
    W = D_inv_sqrt @ eigenvectors.T
    whitened = (W @ centered.T).T
    # Length normalize the whitened version
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    whitened = whitened / (norms + 1e-10)
    # Length normalize the original
    orig_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    orig_normed = embeddings / (orig_norms + 1e-10)
    # Alpha blend
    blended = (1 - alpha) * orig_normed + alpha * whitened
    # Re-normalize
    blend_norms = np.linalg.norm(blended, axis=1, keepdims=True)
    return blended / (blend_norms + 1e-10)


# ═══════════════════════════════════════════════════════════════
# AUTO SPEAKER COUNT (NME Eigengap)
# ═══════════════════════════════════════════════════════════════

def estimate_speakers_from_eigengap(
    affinity: np.ndarray, max_speakers: int = 8
) -> int:
    """
    Estimate speaker count from eigengap of graph Laplacian.
    Improved over naive argmax: picks the HIGHEST k that has a significant
    gap (>= 40% of the max gap).  This prevents underestimation when
    same-gender speakers create a dense affinity block that masks higher-k gaps.

    Park et al. (2020) NME-SC approach, with multi-gap refinement.
    """
    from scipy.sparse.csgraph import laplacian
    L = laplacian(affinity, normed=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    # Look at gaps between eigenvalues 1..max_speakers
    limit = min(max_speakers + 1, len(eigenvalues))
    if limit < 3:
        return 2
    gaps = np.diff(eigenvalues[1:limit])
    if gaps.max() <= 0:
        return 2

    # Standard: largest gap
    max_gap_k = int(np.argmax(gaps)) + 2

    # Multi-gap refinement: find the highest k with a gap >= 40% of max.
    # Prevents eigengap=4 when eigengap=6 also has a strong signal.
    threshold = 0.40 * gaps.max()
    highest_significant_k = max_gap_k
    for i in range(len(gaps) - 1, -1, -1):
        if gaps[i] >= threshold:
            highest_significant_k = i + 2
            break

    best_k = highest_significant_k
    logger.info(
        f"Eigengap analysis: max_gap_k={max_gap_k}, "
        f"highest_significant_k={highest_significant_k}, "
        f"gaps={[round(float(g), 4) for g in gaps]}"
    )
    return min(best_k, max_speakers)


# ═══════════════════════════════════════════════════════════════
# ANCHOR SELECTION
# ═══════════════════════════════════════════════════════════════

def select_anchors_balanced(
    segments: list[dict],
    embeddings: np.ndarray,
    valid_mask: Sequence[bool],
    min_anchors: int = MIN_ANCHORS,
) -> list[int]:
    """
    Select anchors using farthest-first traversal in embedding space.
    Prevents dominant-speaker bias where most anchors come from one person.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Collect candidates: valid segments >= ANCHOR_MIN_DURATION_MS
    candidates = []
    for i in range(len(segments)):
        if not valid_mask[i]:
            continue
        dur_ms = segments[i]["end_ms"] - segments[i]["start_ms"]
        if dur_ms >= ANCHOR_MIN_DURATION_MS:
            candidates.append(i)

    # Relax to 800ms if needed
    if len(candidates) < min_anchors:
        candidates = [
            i for i in range(len(segments))
            if valid_mask[i] and (segments[i]["end_ms"] - segments[i]["start_ms"]) >= ANCHOR_FALLBACK_MS
        ]

    # Use all valid if still not enough
    if len(candidates) < min_anchors:
        candidates = [i for i in range(len(segments)) if valid_mask[i]]

    if len(candidates) <= min_anchors:
        return candidates

    # Farthest-first traversal for diversity
    cand_embs = np.array([embeddings[i] for i in candidates])
    selected = [0]
    target = max(min_anchors, int(len(candidates) * 0.6))

    while len(selected) < target and len(selected) < len(candidates):
        sel_embs = cand_embs[selected]
        # For each unselected, compute max similarity to any selected
        max_sims = []
        for j in range(len(candidates)):
            if j in selected:
                max_sims.append(float("inf"))
                continue
            sims = cosine_similarity(cand_embs[j].reshape(1, -1), sel_embs).max()
            max_sims.append(float(sims))

        # Pick the one LEAST similar to any selected (most diverse)
        next_idx = int(np.argmin(max_sims))
        if max_sims[next_idx] == float("inf"):
            break
        selected.append(next_idx)

    return [candidates[i] for i in selected]


# ═══════════════════════════════════════════════════════════════
# CONVERSATIONAL TEMPORAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════

def build_temporal_constraints(
    segments: list[dict], anchor_indices: list[int]
) -> np.ndarray:
    """
    Build conversational temporal constraint matrix.
    - Adjacent turns (<500ms gap): NEGATIVE boost (likely different speakers)
    - Same-turn continuation (<100ms gap): POSITIVE boost (same speaker)
    - Skip-1 pattern (A-B-A): POSITIVE boost (same speaker returns)
    """
    n = len(anchor_indices)
    constraints = np.zeros((n, n))

    for i in range(n):
        si = segments[anchor_indices[i]]
        for j in range(i + 1, n):
            sj = segments[anchor_indices[j]]
            gap_ms = sj["start_ms"] - si["end_ms"]

            if 0 <= gap_ms < 100:
                # Same-turn continuation — likely same speaker
                constraints[i][j] = 0.15
                constraints[j][i] = 0.15
            elif 100 <= gap_ms < 500:
                # Adjacent turn — likely different speakers
                constraints[i][j] = -0.10
                constraints[j][i] = -0.10

        # Skip-1 pattern: anchor i and anchor i+2 (separated by one turn)
        if i + 2 < n:
            si = segments[anchor_indices[i]]
            sk = segments[anchor_indices[i + 2]]
            gap_total = sk["start_ms"] - si["end_ms"]
            if gap_total < 15000:  # within 15 seconds
                constraints[i][i + 2] = 0.08
                constraints[i + 2][i] = 0.08

    return constraints


# ═══════════════════════════════════════════════════════════════
# CENTROID REFINEMENT
# ═══════════════════════════════════════════════════════════════

def refine_centroids(
    embeddings: np.ndarray,
    labels: list[int],
    num_speakers: int,
    valid_mask: Sequence[bool],
    iterations: int = CENTROID_REFINEMENT_ITERS,
    top_k_ratio: float = CENTROID_TOP_K_RATIO,
    segments: Optional[list[dict]] = None,
) -> list[int]:
    """
    Iteratively refine cluster centroids using only high-confidence members.
    Optionally uses per-segment F0 to penalise reassignment to clusters
    with very different mean pitch (Fix 2: same-gender separation).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    refined = list(labels)

    for iteration in range(iterations):
        centroids = {}
        cluster_f0_means: dict[int, float] = {}
        unique_ids = sorted(set(refined))

        for spk in unique_ids:
            indices = [i for i in range(len(refined)) if refined[i] == spk and valid_mask[i]]
            if len(indices) < 2:
                continue
            spk_embs = np.array([embeddings[i] for i in indices])
            mean_emb = spk_embs.mean(axis=0)
            sims = cosine_similarity(spk_embs, mean_emb.reshape(1, -1)).flatten()
            k = max(2, int(len(sims) * top_k_ratio))
            top_indices = np.argsort(sims)[-k:]
            centroid = spk_embs[top_indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid = centroid / norm
            centroids[spk] = centroid

            # Precompute cluster mean F0 for pitch-aware refinement
            if segments is not None:
                f0s = [
                    segments[j].get("f0_mean", 0)
                    for j in indices
                    if j < len(segments) and segments[j].get("f0_mean", 0) > 0
                ]
                if f0s:
                    cluster_f0_means[spk] = float(np.mean(f0s))

        if not centroids:
            break

        centroid_ids = sorted(centroids.keys())
        centroid_matrix = np.stack([centroids[s] for s in centroid_ids])
        use_pitch = segments is not None and len(cluster_f0_means) >= 2

        for i in range(len(refined)):
            if not valid_mask[i]:
                continue
            sims = cosine_similarity(embeddings[i].reshape(1, -1), centroid_matrix).flatten()

            # Pitch penalty: segment F0 differs from cluster mean by >20Hz →
            # reduce similarity (max ~0.15 penalty at 150Hz diff)
            if use_pitch and segments is not None and i < len(segments):
                seg_f0 = segments[i].get("f0_mean", 0)
                if seg_f0 > 0:
                    for c_idx, cid in enumerate(centroid_ids):
                        if cid in cluster_f0_means:
                            f0_diff = abs(seg_f0 - cluster_f0_means[cid])
                            if f0_diff > 20:
                                sims[c_idx] -= 0.05 * (f0_diff / 50)

            refined[i] = centroid_ids[int(np.argmax(sims))]

    return refined


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING: MERGE TINY CLUSTERS (Fix 3)
# ═══════════════════════════════════════════════════════════════

def merge_tiny_clusters(
    embeddings: np.ndarray,
    labels: list[int],
    valid_mask: Sequence[bool],
    min_size: int = 5,
    merge_threshold: float = 0.55,
) -> list[int]:
    """Merge clusters with < min_size segments into their nearest large neighbor."""
    from sklearn.metrics.pairwise import cosine_similarity

    refined = list(labels)
    unique_labels = sorted(set(labels))
    counts = Counter(labels)

    centroids = {}
    for cid in unique_labels:
        embs = [embeddings[i] for i in range(len(labels)) if labels[i] == cid and valid_mask[i]]
        if embs:
            c = np.mean(embs, axis=0)
            norm = np.linalg.norm(c)
            if norm > 1e-10:
                c = c / norm
            centroids[cid] = c

    for cid in [c for c in unique_labels if counts[c] < min_size and c in centroids]:
        best_target, best_sim = None, -1.0
        for other in unique_labels:
            if other == cid or counts[other] < min_size or other not in centroids:
                continue
            sim = float(cosine_similarity(
                centroids[cid].reshape(1, -1), centroids[other].reshape(1, -1)
            )[0, 0])
            if sim > best_sim:
                best_sim, best_target = sim, other

        if best_target is not None and best_sim > merge_threshold:
            logger.info(
                f"Merging tiny cluster {cid} ({counts[cid]} segs) "
                f"into cluster {best_target} (sim={best_sim:.3f})"
            )
            for i in range(len(refined)):
                if refined[i] == cid:
                    refined[i] = best_target

    return refined


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING: LOW-COHESION CLUSTER RE-SPLITTING (Fix 1)
# ═══════════════════════════════════════════════════════════════

def try_split_low_cohesion_clusters(
    embeddings: np.ndarray,
    labels: list[int],
    segments: list[dict],
    valid_mask: Sequence[bool],
    cohesion_threshold: float = 0.35,
    min_cluster_size: int = 15,
) -> list[int]:
    """
    For clusters with mean internal cosine similarity below threshold,
    split via centroid-distance outlier detection (handles imbalanced splits
    that spectral sub-clustering cannot: e.g. 70:1 → outlier group of 13).

    Guard: only runs when total speaker count >= 3 to protect 2-speaker calls.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 3:
        return labels

    refined = list(labels)
    next_label = max(unique_labels) + 1

    for cluster_id in unique_labels:
        indices = [i for i, lab in enumerate(labels) if lab == cluster_id and valid_mask[i]]
        if len(indices) < min_cluster_size:
            continue

        cluster_embs = np.array([embeddings[i] for i in indices])
        sim_matrix = cosine_similarity(cluster_embs)
        np.fill_diagonal(sim_matrix, 0)
        n_pairs = len(indices) * (len(indices) - 1)
        mean_sim = sim_matrix.sum() / n_pairs if n_pairs > 0 else 1.0

        if mean_sim >= cohesion_threshold:
            continue

        logger.info(
            f"Low-cohesion cluster {cluster_id}: mean_sim={mean_sim:.3f}, "
            f"size={len(indices)} — attempting outlier split"
        )

        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm

        sims_to_centroid = cosine_similarity(
            cluster_embs, centroid.reshape(1, -1)
        ).flatten()

        threshold_pct = np.percentile(sims_to_centroid, 25)
        outlier_mask = sims_to_centroid < threshold_pct
        outlier_indices = [indices[j] for j in range(len(indices)) if outlier_mask[j]]

        if len(outlier_indices) < 5:
            logger.info(f"  Only {len(outlier_indices)} outliers — too few to split")
            continue

        outlier_embs = np.array([embeddings[i] for i in outlier_indices])
        outlier_sim = cosine_similarity(outlier_embs)
        np.fill_diagonal(outlier_sim, 0)
        n_out = len(outlier_indices)
        outlier_cohesion = outlier_sim.sum() / max(1, n_out * (n_out - 1))

        inlier_indices = [indices[j] for j in range(len(indices)) if not outlier_mask[j]]
        inlier_f0s = [segments[i].get("f0_mean", 0) for i in inlier_indices if segments[i].get("f0_mean", 0) > 0]
        outlier_f0s = [segments[i].get("f0_mean", 0) for i in outlier_indices if segments[i].get("f0_mean", 0) > 0]
        f0_diff = abs(np.mean(inlier_f0s) - np.mean(outlier_f0s)) if inlier_f0s and outlier_f0s else 0

        if outlier_cohesion > 0.35 or f0_diff > 10:
            logger.info(
                f"  SPLIT: {len(outlier_indices)} outliers → cluster {next_label} "
                f"(cohesion={outlier_cohesion:.3f}, f0_diff={f0_diff:.1f}Hz)"
            )
            for i in outlier_indices:
                refined[i] = next_label
            next_label += 1
        else:
            logger.info(
                f"  No split — outliers not distinct "
                f"(cohesion={outlier_cohesion:.3f}, f0_diff={f0_diff:.1f}Hz)"
            )

    return refined


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def spectral_diarize(
    segments: list[dict],
    embeddings: np.ndarray,
    num_speakers: int,
    valid_mask: Sequence[bool],
    max_speakers: int = 8,
) -> Optional[list[int]]:
    """
    Two-pass spectral clustering with affinity refinement, embedding whitening,
    conversational constraints, and iterative centroid refinement.

    Args:
        segments: list of dicts with start_ms, end_ms, text
        embeddings: (N, 192) ECAPA-TDNN embeddings
        num_speakers: target number of speakers (0 = auto-detect via NME)
        valid_mask: bool list, True if embedding is usable
        max_speakers: upper bound for auto speaker count

    Returns:
        List of integer labels per segment, or None on failure.
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(segments)
    if n < 2:
        return None

    # ══════════════════════════════════════════════════════
    # PASS 1: Anchor selection + spectral clustering
    # ══════════════════════════════════════════════════════

    # Select diverse anchors via farthest-first traversal
    anchor_indices = select_anchors_balanced(segments, embeddings, valid_mask)

    if len(anchor_indices) < max(2, num_speakers):
        logger.warning(
            f"Only {len(anchor_indices)} anchors, need {num_speakers}. Cannot cluster."
        )
        return None

    n_anchors = len(anchor_indices)
    anchor_embs = np.array([embeddings[i] for i in anchor_indices])

    logger.info(
        f"Spectral clustering v2: {n_anchors} anchors (farthest-first) "
        f"out of {n} segments (target {num_speakers} speakers)"
    )

    # ── Whiten embeddings for same-gender separation ──
    anchor_embs_white = whiten_embeddings(anchor_embs)

    # ── Cosine similarity on whitened embeddings ──
    cos_sim = cosine_similarity(anchor_embs_white)
    cos_sim = np.clip(cos_sim, 0, 1)

    # ── Full affinity refinement pipeline ──
    affinity = refine_affinity(cos_sim, n_anchors)

    # ── Auto speaker count via NME eigengap on CLEAN affinity ──
    # (before temporal constraints which can mask higher-k gaps)
    effective_speakers = num_speakers
    if num_speakers <= 0 or num_speakers > n_anchors:
        effective_speakers = estimate_speakers_from_eigengap(affinity, max_speakers)
        logger.info(f"NME eigengap estimated {effective_speakers} speakers")

    # ── Add conversational temporal constraints ──
    temporal = build_temporal_constraints(segments, anchor_indices)
    affinity = affinity + temporal * TEMPORAL_WEIGHT
    affinity = np.clip(affinity, 0, None)  # no negative affinities
    np.fill_diagonal(affinity, 1.0)  # self-affinity must be 1.0

    if effective_speakers < 2:
        effective_speakers = 2
    if effective_speakers > n_anchors:
        effective_speakers = n_anchors

    # ── Spectral clustering with deterministic label assignment ──
    try:
        sc = SpectralClustering(
            n_clusters=effective_speakers,
            affinity="precomputed",
            assign_labels="discretize",  # Deterministic, robust for near-degenerate spectra
            random_state=42,
        )
        anchor_labels = sc.fit_predict(affinity)
    except Exception as e:
        logger.warning(f"Spectral clustering failed: {e}")
        return None

    counts = Counter(anchor_labels.tolist())
    logger.info(f"Anchor clustering: {dict(counts)}")

    # ══════════════════════════════════════════════════════
    # PASS 2: Assign non-anchor segments with quality-aware weighting
    # ══════════════════════════════════════════════════════

    labels = [0] * n
    anchor_set = set(anchor_indices)
    anchor_mids = np.array([
        (segments[i]["start_ms"] + segments[i]["end_ms"]) / 2000.0
        for i in anchor_indices
    ])

    # Assign anchors
    for idx, ai in enumerate(anchor_indices):
        labels[ai] = int(anchor_labels[idx])

    # Build cluster centroids from anchors
    cluster_centroids = {}
    for cid in range(effective_speakers):
        embs = [anchor_embs[j] for j in range(n_anchors) if anchor_labels[j] == cid]
        if embs:
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid = centroid / norm
            cluster_centroids[cid] = centroid

    # Assign non-anchors with quality-aware weighting
    non_anchors = [i for i in range(n) if i not in anchor_set]

    for i in non_anchors:
        seg_mid = (segments[i]["start_ms"] + segments[i]["end_ms"]) / 2000.0
        emb = embeddings[i]
        emb_valid = valid_mask[i] and np.linalg.norm(emb) > 0.01
        dur_ms = segments[i]["end_ms"] - segments[i]["start_ms"]
        emb_norm = float(np.linalg.norm(emb)) if emb_valid else 0.0

        # Quality-aware weighting (Problem 4 fix)
        if dur_ms >= 1000 and emb_norm > 0.5:
            emb_weight, time_weight = 0.80, 0.20
        elif dur_ms >= 500 and emb_norm > 0.3:
            emb_weight, time_weight = 0.60, 0.40
        else:
            emb_weight, time_weight = 0.20, 0.80

        best_score = -float("inf")
        best_label = 0

        if emb_valid and cluster_centroids:
            for cid, centroid in cluster_centroids.items():
                emb_sim = float(cosine_similarity(
                    emb.reshape(1, -1), centroid.reshape(1, -1)
                )[0, 0])

                # Nearest anchor of this cluster by time
                min_td = float("inf")
                for j in range(n_anchors):
                    if anchor_labels[j] == cid:
                        td = abs(seg_mid - anchor_mids[j])
                        if td < min_td:
                            min_td = td

                time_score = max(0, 1 - min_td / TEMPORAL_WINDOW_SEC) if min_td < TEMPORAL_WINDOW_SEC else 0
                combined = emb_sim * emb_weight + time_score * time_weight
                if combined > best_score:
                    best_score = combined
                    best_label = cid
        else:
            # No valid embedding — purely temporal
            min_dist = float("inf")
            for ai in anchor_indices:
                td = abs(seg_mid - (segments[ai]["start_ms"] + segments[ai]["end_ms"]) / 2000.0)
                if td < min_dist:
                    min_dist = td
                    best_label = labels[ai]

        labels[i] = best_label

    # ── Post-processing pipeline ──
    labels = refine_centroids(embeddings, labels, effective_speakers, valid_mask, segments=segments)
    labels = merge_tiny_clusters(embeddings, labels, valid_mask, min_size=5, merge_threshold=0.55)
    labels = try_split_low_cohesion_clusters(embeddings, labels, segments, valid_mask, cohesion_threshold=0.35)

    # ── Temporal smoothing: fix isolated micro-fragments ──
    for i in range(1, n - 1):
        prev_l, curr_l, next_l = labels[i - 1], labels[i], labels[i + 1]
        if prev_l == next_l and curr_l != prev_l:
            dur_ms = segments[i]["end_ms"] - segments[i]["start_ms"]
            gap_before = segments[i]["start_ms"] - segments[i - 1]["end_ms"]
            gap_after = segments[i + 1]["start_ms"] - segments[i]["end_ms"]
            if dur_ms < 1500 and gap_before < 200 and gap_after < 200:
                labels[i] = prev_l

    # ── Log results ──
    final_counts = Counter(labels)
    logger.info(
        f"Spectral clustering v2 complete: {len(final_counts)} speakers — "
        + ", ".join(f"C{k}: {v}" for k, v in sorted(final_counts.items()))
    )

    return labels
