#!/usr/bin/env python3
"""Run upstream Laya over the same checked-in public-API benchmark fixture."""

import argparse
import json
import os
import platform
import sys
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def timed(device, function):
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def summary(samples):
    ordered = sorted(samples)
    # Match Math.round in the Clojure runner. Python's round uses ties-to-even.
    percentile = lambda fraction: ordered[int(fraction * (len(ordered) - 1) + 0.5)]
    return {
        "samples_ms": samples,
        "min_ms": ordered[0],
        "median_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "max_ms": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laya-source", required=True,
                        help="checkout containing the upstream laya package")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fixture", default=os.path.join(os.path.dirname(__file__),
                                                           "laya_benchmark_case.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    sys.path.insert(0, args.laya_source)
    import laya  # pylint: disable=import-outside-toplevel

    with open(args.fixture, encoding="utf-8") as source:
        fixture = json.load(source)
    state, questions = fixture["state"], fixture["questions"]
    requested = torch.device(args.device)
    synchronize(requested)
    started = time.perf_counter()
    agent = laya.load(args.model_dir, device=args.device)
    synchronize(agent.device)
    load_ms = (time.perf_counter() - started) * 1000.0
    predict = lambda: agent.predict(state, questions)
    first_result, first_ms = timed(agent.device, predict)
    warmup_ms = [timed(agent.device, predict)[1] for _ in range(args.warmup)]
    samples = []
    for _ in range(args.rounds):
        result, elapsed = timed(agent.device, predict)
        if result != first_result:
            raise RuntimeError("Laya benchmark result changed between runs")
        samples.append(elapsed)

    device_name = None
    if agent.device.type == "xpu":
        device_name = torch.xpu.get_device_name(agent.device)
    elif agent.device.type == "cuda":
        device_name = torch.cuda.get_device_name(agent.device)
    print(json.dumps({
        "schema": "pretrained/laya-benchmark-v1",
        "fixture": fixture["id"],
        "implementation": "upstream-laya-torch",
        "device": str(agent.device),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "device_name": device_name,
        },
        "load_ms": load_ms,
        "first_predict_ms": first_ms,
        "warmup_ms": warmup_ms,
        "steady_predict": summary(samples),
        "prediction": first_result,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
