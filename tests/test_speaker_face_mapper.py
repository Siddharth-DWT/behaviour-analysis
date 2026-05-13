from services.video_agent.feature_extractor import SpeakerFaceMapper, WindowFeatures


def _window(face_index: int, start_ms: int, end_ms: int) -> WindowFeatures:
    return WindowFeatures(
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        face_index=face_index,
        frame_count=10,
    )


def test_time_overlap_assignment_is_not_exported_as_identity_link():
    mapper = SpeakerFaceMapper()
    windows = [
        _window(0, 0, 2000),
        _window(1, 0, 2000),
    ]
    diar_segments = [
        {"speaker": "Speaker_0", "start_ms": 0, "end_ms": 2000},
        {"speaker": "Speaker_1", "start_ms": 0, "end_ms": 2000},
    ]

    windows_by_face, lip_sync_scores, face_to_speaker = mapper.assign(
        windows, diar_segments, lip_activity_map=None
    )

    assert set(windows_by_face.keys()) == {"Face_0", "Face_1"}
    assert lip_sync_scores == {}
    assert face_to_speaker == {}


def test_weak_lip_sync_assignment_is_not_exported_as_identity_link():
    mapper = SpeakerFaceMapper()
    windows = [
        _window(0, 0, 2000),
        _window(1, 0, 2000),
    ]
    diar_segments = [
        {"speaker": "Speaker_0", "start_ms": 0, "end_ms": 1000},
        {"speaker": "Speaker_1", "start_ms": 1000, "end_ms": 2000},
    ]
    lip_activity_map = {
        0: [(250, 0.50), (1250, 0.49)],
        1: [(250, 0.49), (1250, 0.50)],
    }

    _, lip_sync_scores, face_to_speaker = mapper.assign(
        windows, diar_segments, lip_activity_map=lip_activity_map
    )

    assert lip_sync_scores == {}
    assert face_to_speaker == {}


def test_positive_lip_sync_assignment_is_exported_as_identity_link():
    mapper = SpeakerFaceMapper()
    windows = [
        _window(0, 0, 2000),
        _window(1, 0, 2000),
    ]
    diar_segments = [
        {"speaker": "Speaker_0", "start_ms": 0, "end_ms": 1000},
        {"speaker": "Speaker_1", "start_ms": 1000, "end_ms": 2000},
    ]
    lip_activity_map = {
        0: [(250, 0.80), (1250, 0.10)],
        1: [(250, 0.10), (1250, 0.80)],
    }

    _, lip_sync_scores, face_to_speaker = mapper.assign(
        windows, diar_segments, lip_activity_map=lip_activity_map
    )

    assert face_to_speaker == {0: "Speaker_0", 1: "Speaker_1"}
    assert lip_sync_scores["Speaker_0"] > 0
    assert lip_sync_scores["Speaker_1"] > 0
