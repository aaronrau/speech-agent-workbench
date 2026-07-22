#!/usr/bin/env python3

import argparse
import os


def prefetch_llama_cpp_model(model_path, repo_id, download_fn=None):
    model_path = str(model_path or "").strip()
    repo_id = str(repo_id or "").strip()
    if not model_path:
        raise ValueError("llama.cpp model path is empty")
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isfile(model_path):
        return model_path
    if not repo_id:
        raise ValueError("llama.cpp model repository is empty")

    if download_fn is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download the llama.cpp model"
            ) from exc
        download_fn = hf_hub_download

    target_dir = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    os.makedirs(target_dir, exist_ok=True)
    print(
        "[model] downloading llama.cpp model "
        f"{repo_id}/{filename} -> {model_path}",
        flush=True,
    )
    downloaded_path = download_fn(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )
    if not os.path.isfile(model_path):
        downloaded_path = os.path.abspath(str(downloaded_path or ""))
        if downloaded_path != model_path or not os.path.isfile(downloaded_path):
            raise RuntimeError(
                "Hugging Face download completed without creating the expected "
                f"model: {model_path}"
            )
    print(f"[model] llama.cpp model ready: {model_path}", flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    prefetch_llama_cpp_model(args.model, args.repo)


if __name__ == "__main__":
    main()
