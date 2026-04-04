"""
One-time script to convert DistilBERT to ONNX format with INT8 quantization.
Run once, then the Language Agent uses the pre-converted model.

Usage:
    python scripts/convert_model_to_onnx.py
    python scripts/convert_model_to_onnx.py --benchmark
"""
import argparse
import os
import shutil
import time
from pathlib import Path


def convert_distilbert():
    """Convert DistilBERT sentiment model to ONNX + INT8."""
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = Path("models/distilbert-onnx")
    quantized_dir = Path("models/distilbert-onnx-int8")

    output_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    print(f"Converting {model_id} to ONNX...")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  Saved ONNX model to {output_dir}")

    # Step 2: INT8 quantization
    print("Applying INT8 quantization...")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(output_dir))
        try:
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
            quantizer.quantize(save_dir=str(quantized_dir), quantization_config=qconfig)
            print("  Applied AVX512-VNNI INT8 quantization")
        except Exception:
            print("  AVX512-VNNI not supported, trying ARM64 config...")
            try:
                qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
                quantizer.quantize(save_dir=str(quantized_dir), quantization_config=qconfig)
                print("  Applied ARM64 INT8 quantization")
            except Exception:
                _fallback_quantize(output_dir, quantized_dir)
    except ImportError:
        _fallback_quantize(output_dir, quantized_dir)

    # Copy tokenizer files to quantized dir if missing
    for f in output_dir.glob("*"):
        if f.suffix in (".json", ".txt") and not (quantized_dir / f.name).exists():
            shutil.copy2(f, quantized_dir / f.name)

    # Size comparison
    orig_size = sum(f.stat().st_size for f in output_dir.glob("*.onnx")) / 1024 / 1024
    quant_size = sum(f.stat().st_size for f in quantized_dir.glob("*.onnx")) / 1024 / 1024
    print(f"  ONNX FP32: {orig_size:.1f} MB")
    print(f"  ONNX INT8: {quant_size:.1f} MB")
    if quant_size > 0:
        print(f"  Compression: {orig_size / quant_size:.1f}x smaller")
    print(f"Done: DistilBERT ONNX INT8 saved to {quantized_dir}")


def _fallback_quantize(output_dir: Path, quantized_dir: Path):
    """Basic dynamic quantization via onnxruntime directly."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    shutil.copytree(str(output_dir), str(quantized_dir), dirs_exist_ok=True)
    onnx_path = str(quantized_dir / "model.onnx")
    quantized_path = str(quantized_dir / "model_quantized.onnx")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
    os.replace(quantized_path, onnx_path)
    print("  Applied basic dynamic INT8 quantization")


def benchmark(model_dir: str, num_samples: int = 100):
    """Quick benchmark of ONNX model inference speed."""
    import numpy as np
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print(f"\nBenchmarking {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForSequenceClassification.from_pretrained(model_dir)

    test_text = "This is a great product and I love using it every day."
    inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True, max_length=128)
    for _ in range(10):
        model(**inputs)

    texts = [
        "I'm really happy with this purchase.",
        "This is terrible, worst experience ever.",
        "The meeting went okay, nothing special.",
        "We need to close this deal by Friday.",
        "I understand your concerns about the pricing.",
    ] * (num_samples // 5)

    start = time.time()
    for text in texts:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        model(**inputs)
    elapsed = time.time() - start

    per_sample = (elapsed / len(texts)) * 1000
    throughput = len(texts) / elapsed
    print(f"  {len(texts)} samples in {elapsed:.2f}s")
    print(f"  {per_sample:.1f} ms/sample")
    print(f"  {throughput:.0f} samples/sec")
    return per_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DistilBERT to ONNX INT8")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after conversion")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    convert_distilbert()

    if args.benchmark:
        print("\n--- PyTorch Baseline ---")
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        texts = ["This is great."] * 100
        pipe(texts[:5])
        start = time.time()
        pipe(texts, batch_size=16)
        pytorch_time = time.time() - start
        pytorch_ms = pytorch_time / len(texts) * 1000
        print(f"  PyTorch DistilBERT: {len(texts)} samples in {pytorch_time:.2f}s ({pytorch_ms:.1f} ms/sample)")

        results = {}
        if Path("models/distilbert-onnx-int8").exists():
            print("\n--- DistilBERT ONNX INT8 ---")
            results["distilbert_onnx_int8"] = benchmark("models/distilbert-onnx-int8")

        if Path("models/distilbert-onnx").exists():
            print("\n--- DistilBERT ONNX FP32 ---")
            results["distilbert_onnx_fp32"] = benchmark("models/distilbert-onnx")

        print(f"\n{'=' * 50}")
        print("SPEEDUP SUMMARY:")
        print(f"  PyTorch DistilBERT:     {pytorch_ms:.1f} ms/sample (baseline)")
        for name, ms in results.items():
            speedup = pytorch_ms / ms if ms > 0 else 0
            print(f"  {name:25s}: {ms:.1f} ms/sample ({speedup:.1f}x faster)")
