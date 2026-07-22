#!/usr/bin/env python3
"""Download and validate the default Parakeet ONNX speech model."""

import argparse
import importlib
import os
import sys


def prefetch_stt_model(config_path, workbench_app):
    config = workbench_app.load_config(config_path)
    model_name = workbench_app.get_parakeet_onnx_model(config)
    quantization = workbench_app.get_parakeet_onnx_quantization(config)

    print(
        "[install] ensuring Parakeet ONNX STT model is downloaded "
        f"(model={model_name}, quantization={quantization})...",
        flush=True,
    )
    workbench_app.load_parakeet_onnx_model(model_name, quantization)
    print("[install] Parakeet ONNX STT model ready.", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    config_path = os.path.abspath(os.path.expanduser(args.config))
    sys.path.insert(0, repo_root)
    workbench_app = importlib.import_module("app")
    prefetch_stt_model(config_path, workbench_app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
