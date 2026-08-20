#!/usr/bin/env python3
"""측정 원본을 모아 사이트가 쓰는 하나의 벤치마크 데이터셋으로 정리한다.

입력은 세 갈래이고 전부 저장소에 함께 커밋되어 있어 결과를 검증할 수 있다.

    benchmarks/logs/trt_*.log              trtexec 원본 출력 (정밀도별 지연시간)
    benchmarks/logs/device_measurements.json  기기에서 직접 잰 값 (eager 기준선, 엔진 크기)
    notebooks/vision_compression_practice.ipynb  CPU 양자화 셀의 실제 출력

출력은 `benchmarks/vision_compression.json` 하나이며, 이 파일이 사이트 차트의
유일한 데이터 소스다.

    python scripts/collect_benchmarks.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "benchmarks" / "logs"
NOTEBOOK = REPO / "notebooks" / "vision_compression_practice.ipynb"
OUTPUT = REPO / "benchmarks" / "vision_compression.json"

MB = 1024 * 1024

# trtexec 요약 블록의 한 줄에서 통계값을 뽑는다.
#   [I] GPU Compute Time: min = 27.8 ms, max = ..., mean = ..., median = ..., percentile(90%) = ...
STAT_LINE = re.compile(
    r"\[I\]\s+(?P<label>[A-Za-z0-9 ]+?):\s+"
    r"min = (?P<min>[\d.]+) ms, max = (?P<max>[\d.]+) ms, "
    r"mean = (?P<mean>[\d.]+) ms, median = (?P<median>[\d.]+) ms, "
    r"percentile\(90%\) = (?P<p90>[\d.]+) ms.*?percentile\(99%\) = (?P<p99>[\d.]+) ms"
)
THROUGHPUT_LINE = re.compile(r"\[I\]\s+Throughput:\s+([\d.]+) qps")

# trtexec 라벨 → 데이터셋에서 쓸 키
WANTED = {
    "Latency": "end_to_end",
    "GPU Compute Time": "gpu_compute",
    "H2D Latency": "host_to_device",
    "D2H Latency": "device_to_host",
    "Enqueue Time": "enqueue",
}


def parse_trtexec(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    stats: dict[str, dict[str, float]] = {}
    for match in STAT_LINE.finditer(text):
        key = WANTED.get(match.group("label").strip())
        if key is None:
            continue
        stats[key] = {
            name: float(match.group(name))
            for name in ("min", "max", "mean", "median", "p90", "p99")
        }

    missing = set(WANTED.values()) - set(stats)
    if missing:
        raise ValueError(f"{path.name}: 통계 항목 누락 {sorted(missing)}")

    throughput = THROUGHPUT_LINE.search(text)
    return {
        "throughput_qps": float(throughput.group(1)) if throughput else None,
        "passed": "PASSED" in text,
        "stats_ms": stats,
        "source_log": path.relative_to(REPO).as_posix(),
    }


def notebook_stdout(path: Path) -> str:
    """노트북의 모든 stdout 출력을 하나의 문자열로 이어붙인다."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                chunks.append("".join(out.get("text", [])))
    return "\n".join(chunks)


def parse_cpu_results(text: str) -> dict:
    """CPU 양자화 비교 셀이 찍은 줄에서 지연시간·크기를 되읽는다.

    형식: `FP32 CPU:  798.4 ms/batch(16)   size 44.6 MB`
    """
    pattern = re.compile(
        r"^(FP32|INT8) CPU:\s+([\d.]+) ms/batch\((\d+)\)\s+size ([\d.]+) MB",
        re.MULTILINE,
    )
    results = {}
    for precision, ms, batch, size in pattern.findall(text):
        results[precision.lower()] = {
            "mean_ms": float(ms),
            "size_mb": float(size),
            "batch": int(batch),
        }
    if {"fp32", "int8"} - set(results):
        raise ValueError(
            "노트북에서 CPU 비교 결과를 찾지 못했다 — 해당 셀을 먼저 실행할 것"
        )
    return results


def build() -> dict:
    device = json.loads((LOG_DIR / "device_measurements.json").read_text(encoding="utf-8"))
    cpu = parse_cpu_results(notebook_stdout(NOTEBOOK))
    engine_bytes = device["engine_bytes"]

    eager = device["pytorch_eager_gpu"]
    baseline_ms = eager["mean_ms"]

    configs = [
        {
            "id": "eager",
            "label": "PyTorch eager",
            "precision": "FP32",
            "runtime": "PyTorch",
            "device": "GPU",
            "mean_ms": baseline_ms,
            "median_ms": eager["median_ms"],
            "size_mb": round(cpu["fp32"]["size_mb"], 1),
            "speedup_vs_eager": 1.0,
            "note": "TensorRT 없이 그대로 실행한 기준선",
        }
    ]

    for precision in ("fp32", "fp16", "int8"):
        trt = parse_trtexec(LOG_DIR / f"trt_{precision}.log")
        gpu = trt["stats_ms"]["gpu_compute"]
        configs.append(
            {
                "id": f"trt_{precision}",
                "label": f"TensorRT {precision.upper()}",
                "precision": precision.upper(),
                "runtime": "TensorRT",
                "device": "GPU",
                "mean_ms": gpu["mean"],
                "median_ms": gpu["median"],
                "size_mb": round(engine_bytes[precision] / MB, 1),
                "speedup_vs_eager": round(baseline_ms / gpu["mean"], 2),
                "latency_ms": trt["stats_ms"]["end_to_end"],
                "breakdown_ms": {
                    "host_to_device": trt["stats_ms"]["host_to_device"]["mean"],
                    "gpu_compute": gpu["mean"],
                    "device_to_host": trt["stats_ms"]["device_to_host"]["mean"],
                },
                "throughput_qps": trt["throughput_qps"],
                "source_log": trt["source_log"],
            }
        )

    return {
        "generated_by": "scripts/collect_benchmarks.py",
        "device": device["device"],
        "workload": {
            "model": "ResNet-18 (torchvision, ImageNet 사전학습)",
            "input": "16 × 3 × 224 × 224",
            "batch": 16,
        },
        "configs": configs,
        "cpu_quantization": {
            "note": "torchvision 양자화 모델은 CPU 전용이라 GPU로 못 올린다.",
            "source": "notebooks/vision_compression_practice.ipynb",
            "batch": cpu["fp32"]["batch"],
            "fp32": cpu["fp32"],
            "int8": cpu["int8"],
            "size_ratio": round(cpu["fp32"]["size_mb"] / cpu["int8"]["size_mb"], 2),
            "speed_ratio": round(cpu["fp32"]["mean_ms"] / cpu["int8"]["mean_ms"], 2),
        },
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        data = build()
    except (OSError, ValueError, KeyError) as exc:
        print(f"[collect_benchmarks] 실패: {exc}")
        return 1

    OUTPUT.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[collect_benchmarks] {OUTPUT.relative_to(REPO)} 갱신")
    for config in data["configs"]:
        print(
            f"  {config['label']:>18s}  {config['mean_ms']:7.2f} ms"
            f"  {config['size_mb']:6.1f} MB  {config['speedup_vs_eager']:>5}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
