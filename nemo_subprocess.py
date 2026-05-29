import os
import sys

import torch
from app import patch_nemo_canary_lora
from nemo.collections.speechlm2.models import SALM


def is_cuda_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "cuda error" in message
        or "illegal instruction" in message
        or "cuda" in message and "error" in message
    )


def load_model(model_name: str, device: str, dtype: torch.dtype) -> SALM:
    patch_nemo_canary_lora()
    model = SALM.from_pretrained(model_name).eval()
    if device == "cuda":
        model = model.to(dtype).to("cuda")
    else:
        model = model.to("cpu")
    return model


def generate_text(model: SALM, wav_path: str, max_new_tokens: int) -> str:
    prompt = f"Transcribe the following: {model.audio_locator_tag}"
    answer_ids = model.generate(
        prompts=[
            [
                {
                    "role": "user",
                    "content": prompt,
                    "audio": [wav_path],
                }
            ]
        ],
        max_new_tokens=max_new_tokens,
    )
    if hasattr(answer_ids, "cpu"):
        answer_ids = answer_ids.cpu()
    return model.tokenizer.ids_to_text(answer_ids[0])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 nemo_subprocess.py <wav_path>")
    wav_path = sys.argv[1]
    model_name = os.environ.get(
        "VOICE_NEMO_MODEL", "nvidia/canary-qwen-2.5b"
    )
    device_pref = os.environ.get("VOICE_NEMO_DEVICE", "auto").strip().lower()
    max_new_tokens = os.environ.get("VOICE_NEMO_MAX_NEW_TOKENS", "128")
    try:
        max_new_tokens = max(1, int(max_new_tokens))
    except (TypeError, ValueError):
        max_new_tokens = 128

    cuda_available = torch.cuda.is_available()
    tried_cuda = False

    if device_pref in ("cuda", "gpu", "auto") and cuda_available:
        tried_cuda = True
        dtype_name = (
            os.environ.get("VOICE_NEMO_DTYPE", "bfloat16").strip().lower()
        )
        torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        try:
            model = load_model(model_name, "cuda", torch_dtype)
            text = generate_text(model, wav_path, max_new_tokens)
            print(f"__TRANSCRIPT__:{text}")
            return
        except Exception as exc:
            if not is_cuda_failure(exc):
                raise
            print(
                f"[nemo_subprocess] cuda failed: {exc}; retrying on cpu",
                file=sys.stderr,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if device_pref in ("cpu", "auto") or not tried_cuda:
        model = load_model(model_name, "cpu", torch.float32)
        text = generate_text(model, wav_path, max_new_tokens)
        print(f"__TRANSCRIPT__:{text}")
        return

    raise SystemExit("No available device for transcription.")


if __name__ == "__main__":
    main()
