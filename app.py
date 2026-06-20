import ast
import bisect
import gc
import hashlib
import http.server
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from glob import glob

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from vosk import KaldiRecognizer, Model
except Exception:
    KaldiRecognizer = None
    Model = None


DEFAULT_CONFIG = {
    "hotkey": "right_shift",
    "sample_rate": 16000,
    "channels": 1,
    "device": None,
    "model_path": "models/vosk-model-small-en-us-0.15",
    "transcribe_backend": "parakeet-onnx",
    "whisper_model": "base",
    "whisper_language": None,
    "whisper_task": "transcribe",
    "whisper_fp16": True,
    "faster_whisper_device": "cpu",
    "faster_whisper_compute_type": "int8",
    "nemo_model": "nvidia/parakeet-tdt-0.6b-v3",
    "nemo_engine": "auto",
    "nemo_use_lhotse": False,
    "nemo_device": "cuda",
    "nemo_dtype": "bfloat16",
    "nemo_max_new_tokens": 128,
    "remote_url": "http://127.0.0.1:8765/transcribe",
    "remote_timeout": 600,
    "remote_wait_timeout": 300,
    "remote_wait_interval": 0.5,
    "sherpa_model_dir": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    "sherpa_model_type": "auto",
    "sherpa_provider": "cpu",
    "parakeet_onnx_model": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "parakeet_onnx_quantization": "int8",
    "fallback_backend": None,
    "transcribe_request_timeout": 120,
    "torch_empty_cache_after_transcribe": None,
    "backend_retry_cooldown": 300,
    "chunked_transcription": True,
    "chunk_transcribe_seconds": 20.0,
    "chunk_pause_window_seconds": 1.0,
    "chunk_vad_frame_ms": 30,
    "chunk_vad_silence_ms": 240,
    "chunk_vad_threshold": 0.012,
    "run_mode": "hotkey",
    "auto_start_speech_ms": 60,
    "auto_pre_roll_seconds": 1.5,
    "auto_poll_seconds": 0.05,
    "auto_max_utterance_seconds": 90.0,
    "auto_pause_hotkey": "ctrl",
    "auto_vad_backend": "rms",
    "auto_sherpa_vad_model": "models/silero_vad.onnx",
    "auto_sherpa_vad_threshold": 0.5,
    "auto_sherpa_vad_min_speech_seconds": 0.05,
    "auto_sherpa_vad_min_silence_seconds": 0.2,
    "auto_sherpa_vad_provider": "cpu",
    "auto_sherpa_vad_num_threads": 1,
    "auto_trigger_word": "agent",
    "auto_trigger_silence_seconds": 2.0,
    "auto_trigger_probe_seconds": 0.5,
    "auto_trigger_min_probe_seconds": 1.0,
    "auto_trigger_probe_window_seconds": 1.5,
    "auto_trigger_arm_timeout_seconds": 8.0,
    "auto_trigger_aliases": ["codex", "code x", "condex"],
    "auto_tmux_switch_session": None,
    "auto_tmux_switch_words": {},
    "auto_enable_terminate_commands": False,
    "auto_tmux_terminate_words": [],
    "auto_tmux_console_log": None,
    "auto_tmux_console_replay": True,
    "auto_tmux_console_retention_seconds": 3600,
    "auto_tmux_console_max_bytes": 1048576,
    "auto_tmux_console_trim_interval_seconds": 60,
    "auto_tmux_console_poll_seconds": 0.05,
    "auto_tmux_console_idle_flush_seconds": 5.0,
    "auto_tmux_summary_enabled": True,
    "auto_tmux_summary_idle_seconds": 5.0,
    "auto_tmux_summary_lines": 50,
    "auto_tmux_summary_max_chars": 8000,
    "auto_tmux_summary_max_new_tokens": 80,
    "agent_completion_log": None,
    "agent_completion_log_retention_seconds": 3600,
    "agent_completion_log_max_bytes": 262144,
    "agent_completion_log_poll_seconds": 0.2,
    "api_enabled": False,
    "api_host": "127.0.0.1",
    "api_port": 8787,
    "api_token": None,
    "tmux_summary_webhook_url": None,
    "tmux_summary_webhook_token": None,
    "tmux_summary_webhook_timeout": 5.0,
    "transcript_correction_backend": "off",
    "transcript_correction_max_new_tokens": 256,
    "transcript_correction_max_chars": 700,
    "transcript_correction_console_log": True,
    "transcript_correction_apply_to_probes": False,
    "transcript_correction_prompt": None,
    "transcript_correction_llama_cpp_path": "llama-cli",
    "transcript_correction_llama_cpp_server_path": "llama-server",
    "transcript_correction_llama_cpp_server_url": "http://127.0.0.1:18087",
    "transcript_correction_llama_cpp_server_autostart": True,
    "transcript_correction_llama_cpp_server_startup_timeout": 60.0,
    "transcript_correction_llama_cpp_model": (
        "models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"
    ),
    "transcript_correction_llama_cpp_gpu_layers": 99,
    "transcript_correction_llama_cpp_timeout": 20.0,
    "submit_enter_delay_seconds": 0.5,
    "transcript_history_path": "transcript-history.txt",
}

WTYPE_AVAILABLE = True
YDOTOOL_AVAILABLE = True
PASTE_IN_PROGRESS = threading.Event()
PASTE_DEBUG_DEFAULT = False
PASTE_DELAY_DEFAULT = 0.1
PASTE_MODE_DEFAULT = "type"
AUTO_REFOCUS_DELAY_DEFAULT = 0.05
WHISPER_MODEL_CACHE = {}
FASTER_WHISPER_MODEL_CACHE = {}
NEMO_MODEL_CACHE = {}
SHERPA_MODEL_CACHE = {}
PARAKEET_ONNX_MODEL_CACHE = {}
LLAMA_CPP_CORRECTOR_LOCK = threading.Lock()
TRANSCRIPT_CORRECTION_FAILURES = set()
LLAMA_CPP_SERVER_PROCESS = None
TMUX_RECENT_COMMANDS = {}
ALLOWED_TRANSCRIPT_SYMBOLS = {".", "+"}


def normalize_hotkey_name(name):
    if not name:
        return "right_shift"
    return str(name).strip().lower()


def log_nemo_debug(message):
    if os.environ.get("VOICE_NEMO_DEBUG") == "1":
        print(f"[transcribe] nemo debug: {message}", file=sys.stderr)


def is_nemo_cuda_failure(exc):
    message = str(exc).lower()
    return (
        "cuda error" in message
        or "illegal instruction" in message
        or "unspecified launch failure" in message
        or "launch failure" in message and "cuda" in message
    )


def sanitize_transcript_text(text):
    if text is None:
        return ""
    cleaned = []
    for ch in str(text):
        if ch.isalnum() or ch in ALLOWED_TRANSCRIPT_SYMBOLS:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def split_trailing_submit_command(text):
    if not text:
        return "", False
    stripped = str(text).rstrip()
    submit_word = os.environ.get("VOICE_SUBMIT_WORD", "send").strip() or "send"
    match = re.search(rf"(?i)(?:^|\s){re.escape(submit_word)}[.+]*$", stripped)
    if not match:
        return text, False
    return stripped[: match.start()].rstrip(), True


def extract_text_after_trigger_word(text, trigger_word, aliases=None):
    trigger_words = [str(trigger_word or "").strip()]
    trigger_words.extend(parse_word_list(aliases))
    trigger_words = [word for word in trigger_words if word]
    if not text or not trigger_words:
        return None
    stripped = str(text).strip()
    for trigger in sorted(trigger_words, key=len, reverse=True):
        pattern = rf"(?i)^{re.escape(trigger)}[.+]*(?:\s|$)"
        match = re.search(pattern, stripped)
        if match:
            return stripped[match.end() :].strip()
    return None


def make_auto_trigger_session():
    return {
        "armed": False,
        "clicked": False,
        "focus_failed": False,
        "armed_at": None,
        "source": None,
    }


def arm_auto_trigger_session(session, source, now=None, focus_success=True):
    focused = bool(focus_success)
    session["armed"] = True
    session["clicked"] = focused
    session["focus_failed"] = not focused
    session["armed_at"] = time.monotonic() if now is None else now
    session["source"] = source


def reset_auto_trigger_session(session):
    session["armed"] = False
    session["clicked"] = False
    session["focus_failed"] = False
    session["armed_at"] = None
    session["source"] = None


def is_auto_trigger_session_armed(session, timeout_seconds, now=None):
    if not session.get("armed"):
        return False
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0.0:
        reset_auto_trigger_session(session)
        return False
    armed_at = session.get("armed_at")
    current = time.monotonic() if now is None else now
    if armed_at is None or current - float(armed_at) > timeout:
        reset_auto_trigger_session(session)
        return False
    return True


def parse_word_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = str(value).split(",")
    words = []
    seen = set()
    for candidate in candidates:
        word = str(candidate or "").strip()
        key = word.lower()
        if word and key not in seen:
            words.append(word)
            seen.add(key)
    return words


def normalize_voice_command_text(text):
    words = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    return " ".join(words)


COMMON_CODING_TERM_CORRECTIONS = (
    (re.compile(r"(?i)\bcon\s*dex\b"), "Codex"),
    (re.compile(r"(?i)\bcode\s*x\b"), "Codex"),
    (re.compile(r"(?i)\bcodec\b"), "Codex"),
    (re.compile(r"(?i)\bkodex\b"), "Codex"),
    (re.compile(r"(?i)\bt\s*mux\b"), "tmux"),
    (re.compile(r"(?i)\btea\s*mux\b"), "tmux"),
    (re.compile(r"(?i)\bgit\s*hub\b"), "GitHub"),
    (re.compile(r"(?i)\blength\s+view\b"), "Langfuse"),
    (re.compile(r"(?i)\blang\s+fuse\b"), "Langfuse"),
    (re.compile(r"(?i)\bland\s+fuse\b"), "Langfuse"),
    (re.compile(r"(?i)\blangfuse\b"), "Langfuse"),
    (re.compile(r"(?i)\byaws\b"), "EVALS"),
    (re.compile(r"(?i)\be\s*values\b"), "EVALS"),
    (re.compile(r"(?i)\be\s+vals\b"), "EVALS"),
    (re.compile(r"(?i)\bevals\b"), "EVALS"),
    (
        re.compile(r"(?i)\b(?:w)?hen\s+all\s+the\s+chains\s+push(?:ed)?\s+to\s+death\b"),
        "did all the changes get pushed to dev",
    ),
    (
        re.compile(r"(?i)\bdid\s+all\s+the\s+change\s+got\s+pushed\s+to\s+dev\b"),
        "did all the changes get pushed to dev",
    ),
)


COMMAND_TOKEN_ALIASES = {
    "0": ("zero",),
    "1": ("one", "won"),
    "2": ("two", "to", "too"),
    "3": ("three", "tree", "free"),
    "4": ("four", "for"),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "one": ("1", "won"),
    "two": ("2", "to", "too"),
    "three": ("3", "tree", "free"),
    "four": ("4", "for"),
    "codex": ("code x", "condex", "codec", "kodex"),
    "brock": ("block", "broc"),
    "flux": ("flex", "flax"),
    "pike": ("pipe", "pyke"),
    "wolf": ("wulf", "wolfe"),
}


def correct_common_coding_terms(text):
    corrected = str(text or "")
    for pattern, replacement in COMMON_CODING_TERM_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


def build_command_text_aliases(command_text):
    normalized = normalize_voice_command_text(command_text)
    if not normalized:
        return []
    aliases = {normalized}
    tokens = normalized.split()
    for index, token in enumerate(tokens):
        for alias in COMMAND_TOKEN_ALIASES.get(token, ()):
            alias_tokens = tokens[:index] + alias.split() + tokens[index + 1 :]
            aliases.add(" ".join(alias_tokens))
    return sorted(aliases, key=lambda value: (-len(value.split()), value))


def build_terminate_command_text_aliases(command_text):
    aliases = set(build_command_text_aliases(command_text))
    for alias in list(aliases):
        tokens = alias.split()
        if len(tokens) < 2:
            continue
        if tokens[-2] not in ("terminate", "terminates"):
            continue
        if tokens[-1] not in ("session", "sessions"):
            continue
        prefix = tokens[:-2]
        for verb in ("terminate", "terminates"):
            for noun in ("session", "sessions"):
                aliases.add(" ".join(prefix + [verb, noun]))
    return sorted(aliases, key=lambda value: (-len(value.split()), value))


def strip_voice_attention_words(words):
    attention_words = {"hey", "hi", "ok", "okay", "please"}
    index = 0
    while index < len(words) and words[index] in attention_words:
        index += 1
    return words[index:], index


def parse_config_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "none", "null", "off", ""):
        return False
    return None


def terminate_commands_enabled(config):
    value = os.environ.get("VOICE_AUTO_ENABLE_TERMINATE_COMMANDS")
    if value is None:
        value = config.get(
            "auto_enable_terminate_commands",
            DEFAULT_CONFIG["auto_enable_terminate_commands"],
        )
    return bool(parse_config_bool(value))


def parse_key_value_list(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    else:
        chunks = str(value).split(",")
        items = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                key, item_value = chunk.split("=", 1)
            elif ":" in chunk:
                key, item_value = chunk.split(":", 1)
            else:
                key = chunk
                item_value = chunk
            items.append((key, item_value))

    parsed = {}
    for key, item_value in items:
        key = str(key or "").strip()
        item_value = str(item_value or "").strip()
        if key and item_value:
            parsed[key] = item_value
    return parsed


def build_auto_tmux_switch_commands(config):
    session = get_config_string(
        config,
        "VOICE_AUTO_TMUX_SESSION",
        "auto_tmux_switch_session",
        None,
    )
    if not session:
        return {}

    switch_value = os.environ.get("VOICE_AUTO_TMUX_SWITCHES")
    if switch_value is None:
        switch_value = config.get("auto_tmux_switch_words", {})

    switches = parse_key_value_list(switch_value)
    commands = {}
    for spoken_name, target_name in switches.items():
        normalized = normalize_voice_command_text(spoken_name)
        if not normalized:
            continue
        target_name = str(target_name or "").strip()
        tmux_send_target = None
        if target_name.startswith("pane:"):
            pane_target = target_name[5:]
            tmux_send_target = pane_target
            if pane_target.startswith("%"):
                argv = ["tmux", "select-pane", "-t", pane_target]
            else:
                window_target = pane_target.rsplit(".", 1)[0]
                argv = [
                    "tmux",
                    "select-window",
                    "-t",
                    window_target,
                    ";",
                    "select-pane",
                    "-t",
                    pane_target,
                ]
        else:
            tmux_send_target = f"{session}:{target_name}"
            argv = ["tmux", "select-window", "-t", tmux_send_target]
        command = {
            "label": spoken_name,
            "argv": argv,
            "tmux_send_target": tmux_send_target,
        }
        for alias in build_command_text_aliases(normalized):
            commands.setdefault(alias, command)

    if terminate_commands_enabled(config):
        terminate_value = os.environ.get("VOICE_AUTO_TMUX_TERMINATE_WORDS")
        if terminate_value is None:
            terminate_value = config.get(
                "auto_tmux_terminate_words",
                DEFAULT_CONFIG["auto_tmux_terminate_words"],
            )
        for spoken_name in parse_word_list(terminate_value):
            normalized = normalize_voice_command_text(spoken_name)
            if not normalized:
                continue
            command = {
                "label": spoken_name,
                "argv": ["tmux", "kill-session", "-t", session],
                "success_message": f"[auto] terminating tmux session: {session}",
                "exit_after": True,
                "allow_prefix": False,
                "requires_explicit_audio": True,
            }
            for alias in build_terminate_command_text_aliases(normalized):
                commands.setdefault(alias, command)
    return commands


def normalize_auto_command_phrase(text):
    phrase_words = normalize_voice_command_text(text).split()
    phrase_words, _offset = strip_voice_attention_words(phrase_words)
    return " ".join(phrase_words)


def strip_control_command_filler(words):
    filler_words = {"please", "now"}
    stripped = list(words)
    while stripped and stripped[-1] in filler_words:
        stripped.pop()
    return stripped


def match_exact_only_auto_shell_command_with_filler(phrase_words, commands):
    for command_text, command in commands.items():
        if command.get("allow_prefix") is not False:
            continue
        command_tokens = command_text.split()
        if not command_tokens:
            continue
        if phrase_words[: len(command_tokens)] != command_tokens:
            continue
        trailing_words = phrase_words[len(command_tokens) :]
        if strip_control_command_filler(trailing_words):
            continue
        return command
    return None


def match_auto_shell_command(text, commands):
    if not commands:
        return None

    phrase = normalize_auto_command_phrase(text)
    if phrase in commands:
        return commands[phrase]
    phrase_words = phrase.split()
    exact_with_filler = match_exact_only_auto_shell_command_with_filler(
        phrase_words,
        commands,
    )
    if exact_with_filler is not None:
        return exact_with_filler

    prefixes = (
        "switch to",
        "switch",
        "go to",
        "focus",
        "show",
        "select",
        "open",
    )
    suffixes = ("terminal", "tab", "window", "agent")
    for prefix in prefixes:
        if phrase == prefix or not phrase.startswith(prefix + " "):
            continue
        target = phrase[len(prefix) + 1 :].strip()
        if target.startswith("the "):
            target = target[4:].strip()
        if target in commands:
            return commands[target]
        for suffix in suffixes:
            if target.endswith(" " + suffix):
                bare_target = target[: -len(suffix) - 1].strip()
                if bare_target in commands:
                    return commands[bare_target]
    return None


def safe_match_auto_shell_command(text, commands):
    try:
        return match_auto_shell_command(text, commands)
    except Exception as exc:
        print(f"[auto] command parse failed: {exc}", file=sys.stderr)
        return None


def match_auto_shell_command_prefix(text, commands):
    if not commands:
        return None

    original = str(text or "").strip()
    if not original:
        return None

    tokens = list(re.finditer(r"[A-Za-z0-9]+", original))
    if not tokens:
        return None

    spoken_tokens = [token.group(0).lower() for token in tokens]
    spoken_tokens, start_index = strip_voice_attention_words(spoken_tokens)
    phrase = " ".join(spoken_tokens)
    if phrase in commands and commands[phrase].get("allow_prefix") is False:
        return None

    matches = []
    for command_text, command in commands.items():
        if command.get("allow_prefix") is False:
            continue
        command_tokens = command_text.split()
        if not command_tokens:
            continue
        if spoken_tokens[: len(command_tokens)] != command_tokens:
            continue
        end = tokens[start_index + len(command_tokens) - 1].end()
        remainder = original[end:].lstrip(" .,+:-").strip()
        matches.append((len(command_tokens), command, remainder))

    if not matches:
        return None
    _length, command, remainder = max(matches, key=lambda item: item[0])
    return command, remainder


def safe_match_auto_shell_command_prefix(text, commands):
    try:
        return match_auto_shell_command_prefix(text, commands)
    except Exception as exc:
        print(f"[auto] command-prefix parse failed: {exc}", file=sys.stderr)
        return None


def auto_shell_command_has_explicit_audio(command, correction):
    if not command or not command.get("requires_explicit_audio"):
        return True
    if not correction:
        return False
    aliases = set(
        build_terminate_command_text_aliases(
            normalize_voice_command_text(command.get("label"))
        )
    )
    if not aliases:
        return False
    for key in ("raw_transcript", "pre_llm_transcript"):
        phrase_words = normalize_auto_command_phrase(correction.get(key)).split()
        phrase = " ".join(strip_control_command_filler(phrase_words))
        if phrase in aliases:
            return True
    return False


def auto_shell_command_allowed(command, correction):
    try:
        if auto_shell_command_has_explicit_audio(command, correction):
            return True
    except Exception as exc:
        print(f"[auto] command safety check failed: {exc}", file=sys.stderr)
        return False
    label = command.get("label") if command else "unknown"
    print(
        "[auto] ignored control command without explicit ASR transcript: "
        f"{label}",
        file=sys.stderr,
    )
    return False


def format_colored_detection_words():
    color_value = os.environ.get("VOICE_AUTO_COLOR_NAMES", "1").strip().lower()
    display_words = parse_key_value_list(os.environ.get("VOICE_AUTO_DISPLAY_WORDS"))
    if not display_words:
        display_words = {
            "agent": "32",
            "agent two": "94",
            "agent three": "38;5;208",
            "voice": "90",
        }
    if color_value in ("0", "false", "no", "off"):
        return ", ".join(display_words.keys())
    return ", ".join(
        f"\033[{code}m{name}\033[0m" for name, code in display_words.items()
    )


def compact_history_text(value):
    return " ".join(str(value or "").split())


def append_transcript_history(text, history_path, correction=None):
    if not history_path:
        return False
    transcript = str(text or "")
    if not transcript.strip():
        return False
    path = os.path.expanduser(str(history_path))
    directory = os.path.dirname(os.path.abspath(path))
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload = transcript
        if correction:
            payload_record = {"text": compact_history_text(transcript)}
            for key in (
                "raw_transcript",
                "pre_llm_transcript",
                "corrected_transcript",
                "correction_backend",
                "model_output",
                "model_accepted",
                "model_skipped",
                "fallback_reason",
            ):
                if key in correction and correction[key] is not None:
                    value = correction[key]
                    if isinstance(value, str):
                        value = compact_history_text(value)
                    payload_record[key] = value
            payload = json.dumps(payload_record, sort_keys=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{payload}\n")
    except OSError as exc:
        print(f"[history] failed to append transcript: {exc}", file=sys.stderr)
        return False
    return True


def mix_audio_to_mono(samples):
    if samples is None:
        return np.empty(0, dtype=np.float32)
    audio = np.asarray(samples)
    if audio.size == 0:
        return np.empty(0, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / 32768.0
    return audio


def choose_chunk_end_sample(
    samples,
    sample_rate,
    target_seconds,
    pause_window_seconds,
    frame_ms,
    silence_ms,
    silence_threshold,
):
    audio = mix_audio_to_mono(samples)
    total_samples = len(audio)
    target_samples = int(max(0.0, target_seconds) * sample_rate)
    if total_samples <= target_samples:
        return None

    window_start = max(
        0,
        int(max(0.0, target_seconds - pause_window_seconds) * sample_rate),
    )
    window_end = min(
        total_samples,
        int((target_seconds + max(0.0, pause_window_seconds)) * sample_rate),
    )
    if window_end <= window_start:
        return min(target_samples, total_samples)

    frame_size = max(1, int(sample_rate * max(frame_ms, 1) / 1000.0))
    silence_frames = max(1, int(math.ceil(max(silence_ms, 1) / max(frame_ms, 1))))
    starts = list(range(0, max(total_samples - frame_size + 1, 1), frame_size))
    if not starts:
        return min(target_samples, total_samples)

    energies = []
    for start in starts:
        frame = audio[start : start + frame_size]
        if frame.size == 0:
            energies.append(0.0)
            continue
        energies.append(float(np.sqrt(np.mean(np.square(frame)))))

    noise_floor = float(np.percentile(energies, 20)) if energies else 0.0
    threshold = max(float(silence_threshold), noise_floor * 3.0)
    candidates = []
    run_start = None

    for index in range(len(energies) + 1):
        is_silent = index < len(energies) and energies[index] <= threshold
        if is_silent:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue
        run_length = index - run_start
        if run_length >= silence_frames:
            silence_start = starts[run_start]
            if index < len(starts):
                silence_end = min(total_samples, starts[index] + frame_size)
            else:
                silence_end = total_samples
            boundary = (silence_start + silence_end) // 2
            if window_start <= boundary <= window_end:
                candidates.append(boundary)
        run_start = None

    if candidates:
        return min(candidates, key=lambda value: abs(value - target_samples))
    if total_samples >= window_end:
        return min(target_samples, total_samples)
    return None


def normalize_backend(name):
    if not name:
        return "vosk"
    value = str(name).strip().lower()
    if value in ("parakey", "parakeet"):
        return "parakeet-onnx"
    return value


def get_fallback_backend(config):
    override = os.environ.get("VOICE_FALLBACK_BACKEND")
    if override is None:
        value = config.get("fallback_backend")
    else:
        value = override
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("", "0", "false", "no", "none", "null", "off"):
        return None
    return normalize_backend(normalized)


def log_backend_fallback(source_backend, target_backend, reason):
    print(
        f"[transcribe] backend '{source_backend}' failed; "
        f"falling back to '{target_backend}': {reason}",
        file=sys.stderr,
    )


def get_hotkey_backend():
    override = os.environ.get("VOICE_HOTKEY_BACKEND")
    if override:
        override = override.strip().lower()
        if override in ("evdev", "pynput"):
            return override
        raise ValueError(
            "VOICE_HOTKEY_BACKEND must be 'evdev' or 'pynput'."
        )
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
        return "evdev"
    return "pynput"


def parse_hotkey_pynput(name):
    from pynput import keyboard as pynput_keyboard

    name = normalize_hotkey_name(name)
    hotkey_map = {
        "right_shift": pynput_keyboard.Key.shift_r,
        "left_shift": pynput_keyboard.Key.shift_l,
        "right_ctrl": pynput_keyboard.Key.ctrl_r,
        "left_ctrl": pynput_keyboard.Key.ctrl_l,
        "right_alt": pynput_keyboard.Key.alt_r,
        "left_alt": pynput_keyboard.Key.alt_l,
        "right_cmd": pynput_keyboard.Key.cmd_r,
        "left_cmd": pynput_keyboard.Key.cmd_l,
        "space": pynput_keyboard.Key.space,
        "tab": pynput_keyboard.Key.tab,
        "enter": pynput_keyboard.Key.enter,
        "esc": pynput_keyboard.Key.esc,
        "escape": pynput_keyboard.Key.esc,
    }
    if name in hotkey_map:
        return hotkey_map[name]
    if hasattr(pynput_keyboard.Key, name):
        return getattr(pynput_keyboard.Key, name)
    if len(name) == 1:
        return pynput_keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unsupported hotkey '{name}'.")


def parse_auto_pause_hotkey_pynput(name):
    from pynput import keyboard as pynput_keyboard

    name = normalize_hotkey_name(name)
    if name in ("ctrl", "control"):
        return {pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r}
    return {parse_hotkey_pynput(name)}


def parse_hotkey_evdev(name):
    from evdev import ecodes

    name = normalize_hotkey_name(name)
    hotkey_map = {
        "right_shift": ecodes.KEY_RIGHTSHIFT,
        "left_shift": ecodes.KEY_LEFTSHIFT,
        "right_ctrl": ecodes.KEY_RIGHTCTRL,
        "left_ctrl": ecodes.KEY_LEFTCTRL,
        "right_alt": ecodes.KEY_RIGHTALT,
        "left_alt": ecodes.KEY_LEFTALT,
        "right_cmd": ecodes.KEY_RIGHTMETA,
        "left_cmd": ecodes.KEY_LEFTMETA,
        "space": ecodes.KEY_SPACE,
        "tab": ecodes.KEY_TAB,
        "enter": ecodes.KEY_ENTER,
        "esc": ecodes.KEY_ESC,
        "escape": ecodes.KEY_ESC,
    }
    if name in hotkey_map:
        return hotkey_map[name]
    if len(name) == 1:
        if name.isalpha():
            key_name = f"KEY_{name.upper()}"
        elif name.isdigit():
            key_name = f"KEY_{name}"
        else:
            key_name = None
        if key_name:
            code = ecodes.ecodes.get(key_name)
            if code is not None:
                return code
    if name.startswith("f") and name[1:].isdigit():
        key_name = f"KEY_F{name[1:]}"
        code = ecodes.ecodes.get(key_name)
        if code is not None:
            return code
    if name.startswith("key_"):
        code = ecodes.ecodes.get(name.upper())
        if code is not None:
            return code
    raise ValueError(f"Unsupported hotkey '{name}'.")


def parse_auto_pause_hotkey_evdev(name):
    from evdev import ecodes

    name = normalize_hotkey_name(name)
    if name in ("ctrl", "control"):
        return {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}
    return {parse_hotkey_evdev(name)}


class Recorder:
    def __init__(self, sample_rate, channels, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._frames = []
        self._frame_offsets = []
        self._level = 0.0
        self._sample_count = 0
        self._stream = None
        self._lock = threading.Lock()
        self._recording_id = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        samples = indata.astype(np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        level = min(1.0, peak / 32768.0)
        with self._lock:
            self._frame_offsets.append(self._sample_count)
            self._frames.append(indata.copy())
            self._sample_count += len(indata)
            self._level = level

    def start(self):
        if sd is None:
            raise RuntimeError(
                "sounddevice not installed; install it to record audio."
            )
        with self._lock:
            self._recording_id += 1
            self._frames = []
            self._frame_offsets = []
            self._sample_count = 0
            recording_id = self._recording_id
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()
        return recording_id

    def stop(self, return_samples=True):
        if self._stream is None:
            return np.empty((0, self.channels), dtype=np.int16)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not return_samples:
            return np.empty((0, self.channels), dtype=np.int16)
        with self._lock:
            if not self._frames:
                return np.empty((0, self.channels), dtype=np.int16)
            return np.concatenate(self._frames, axis=0)

    def get_level(self):
        with self._lock:
            return self._level

    def get_total_samples(self):
        with self._lock:
            return self._sample_count

    def clear_if_idle(self, recording_id):
        with self._lock:
            if self._stream is not None or self._recording_id != recording_id:
                return False
            self._frames = []
            self._frame_offsets = []
            self._sample_count = 0
            self._level = 0.0
            return True

    def discard_before(self, sample_index):
        with self._lock:
            if not self._frames:
                return
            keep_from = 0
            for index, offset in enumerate(self._frame_offsets):
                frame_end = offset + len(self._frames[index])
                if frame_end > sample_index:
                    break
                keep_from = index + 1
            if keep_from <= 0:
                return
            self._frames = self._frames[keep_from:]
            self._frame_offsets = self._frame_offsets[keep_from:]

    def get_samples_since(self, start_sample):
        with self._lock:
            total = self._sample_count
            if not self._frames or start_sample >= total:
                return (
                    np.empty((0, self.channels), dtype=np.int16),
                    total,
                )
            frame_index = bisect.bisect_right(
                self._frame_offsets, start_sample
            ) - 1
            frame_index = max(frame_index, 0)
            first_offset = self._frame_offsets[frame_index]
            offset_in_frame = max(0, start_sample - first_offset)
            samples = np.concatenate(self._frames[frame_index:], axis=0)
            if offset_in_frame:
                samples = samples[offset_in_frame:]
            return samples, total


def load_config(path):
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    merged = DEFAULT_CONFIG.copy()
    merged.update(data or {})
    return merged


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def signal_voice_ready(run_mode, transcribe_backend, model_label):
    path = os.environ.get("VOICE_READY_FILE", "").strip()
    if not path:
        return

    payload = {
        "pid": os.getpid(),
        "run_mode": run_mode,
        "transcribe_backend": transcribe_backend,
        "model": model_label,
        "ready_at": time.time(),
    }
    directory = os.path.dirname(path)
    tmp_path = None
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_dir = directory or "."
        prefix = f".{os.path.basename(path)}."
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=tmp_dir,
            prefix=prefix,
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print(
            f"[ready] unable to write ready file '{path}': {exc}",
            file=sys.stderr,
        )


def prompt_change_saved_value(label, description, current_value):
    if not config_prompts_enabled():
        return False
    print(f"{label}{description}: {current_value}")
    while True:
        choice = input(f"{label}Change {description}? [y/N] ").strip().lower()
        if choice in ("", "n", "no"):
            return False
        if choice in ("y", "yes"):
            return True
        print("Enter y or n.")


def config_prompts_enabled():
    value = os.environ.get("VOICE_CONFIG_PROMPT")
    if value is None:
        value = os.environ.get("VOICE_DEVICE_PROMPT")
    if value is None:
        return True
    return str(value).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "none",
        "null",
        "off",
    )


def format_input_device(device):
    rate = device["rate"]
    rate_display = f"{int(rate)} Hz" if rate else "unknown rate"
    return (
        f"[{device['index']}] {device['name']} - {device['hostapi']} "
        f"({device['channels']} ch, {rate_display})"
    )


def save_input_device_selection(config, device):
    config["device"] = device["index"]
    config["device_name"] = device["name"]
    config["device_hostapi"] = device["hostapi"]


def has_saved_input_preference(config):
    return bool(
        str(config.get("device_name") or "").strip()
        or str(config.get("device_hostapi") or "").strip()
        or config.get("device") is not None
    )


def resolve_saved_input_device(config, devices):
    saved_name = str(config.get("device_name") or "").strip()
    saved_hostapi = str(config.get("device_hostapi") or "").strip()
    if saved_name:
        for device in devices:
            if device["name"] != saved_name:
                continue
            if saved_hostapi and device["hostapi"] != saved_hostapi:
                continue
            config["device"] = device["index"]
            if not saved_hostapi:
                config["device_hostapi"] = device["hostapi"]
            return device
        if not saved_hostapi:
            name_matches = [
                device for device in devices if device["name"] == saved_name
            ]
            if len(name_matches) == 1:
                matched = name_matches[0]
                save_input_device_selection(config, matched)
                return matched

    saved_device = config.get("device")
    if isinstance(saved_device, str):
        legacy_name = saved_device.strip()
        if legacy_name and not legacy_name.isdigit():
            name_matches = [
                device for device in devices if device["name"] == legacy_name
            ]
            if len(name_matches) == 1:
                matched = name_matches[0]
                save_input_device_selection(config, matched)
                return matched

    for device in devices:
        if device["index"] == saved_device:
            if not saved_name:
                save_input_device_selection(config, device)
            return device
    return None


def resolve_saved_evdev_device(config, devices):
    saved_name = str(config.get("evdev_device_name") or "").strip()
    if saved_name:
        return next(
            (device for device in devices if device.name == saved_name), None
        )

    saved_path = str(config.get("evdev_device_path") or "").strip()
    if not saved_path:
        return None

    saved_device = next(
        (device for device in devices if device.path == saved_path), None
    )
    if saved_device is not None:
        config["evdev_device_name"] = saved_device.name
    config.pop("evdev_device_path", None)
    return saved_device


def list_evdev_keyboards():
    import evdev
    from evdev import ecodes

    devices = []
    permission_errors = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
        except PermissionError:
            permission_errors.append(path)
            continue
        except OSError:
            continue
        try:
            capabilities = device.capabilities()
        except OSError:
            device.close()
            continue
        if ecodes.EV_KEY in capabilities:
            devices.append(device)
        else:
            device.close()
    return devices, permission_errors


def find_evdev_devices(hotkey_code):
    import evdev
    from evdev import ecodes

    devices = []
    permission_errors = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
        except PermissionError:
            permission_errors.append(path)
            continue
        except OSError:
            continue
        try:
            capabilities = device.capabilities()
        except OSError:
            device.close()
            continue
        keys = capabilities.get(ecodes.EV_KEY, [])
        if hotkey_code in keys:
            devices.append(device)
        else:
            device.close()
    return devices, permission_errors


def device_supports_hotkey(device, hotkey_code):
    from evdev import ecodes

    try:
        capabilities = device.capabilities()
    except OSError:
        return False
    keys = capabilities.get(ecodes.EV_KEY, [])
    return hotkey_code in keys


def device_supports_any_hotkey(device, hotkey_codes):
    from evdev import ecodes

    try:
        capabilities = device.capabilities()
    except OSError:
        return False
    keys = capabilities.get(ecodes.EV_KEY, [])
    return any(code in keys for code in hotkey_codes)


def drain_evdev_events(device):
    # Clear queued key events so menu input doesn't become the hotkey.
    try:
        device.set_nonblocking(True)
    except AttributeError:
        pass
    try:
        while True:
            event = device.read_one()
            if event is None:
                break
    except (BlockingIOError, OSError):
        pass
    finally:
        try:
            device.set_nonblocking(False)
        except AttributeError:
            pass


def wait_for_evdev_idle(device, quiet_time=0.25, timeout=2.0):
    # Wait for a short quiet period to avoid capturing prior menu input.
    try:
        device.set_nonblocking(True)
    except AttributeError:
        pass
    start = time.monotonic()
    last_event = start
    try:
        while True:
            now = time.monotonic()
            if now - last_event >= quiet_time:
                break
            if now - start >= timeout:
                break
            event = device.read_one()
            if event is None:
                time.sleep(0.01)
                continue
            last_event = now
    except (BlockingIOError, OSError):
        pass
    finally:
        try:
            device.set_nonblocking(False)
        except AttributeError:
            pass


def hotkey_name_from_evdev(key_name):
    if not key_name:
        return None
    if key_name == "KEY_RIGHTSHIFT":
        return "right_shift"
    if key_name == "KEY_LEFTSHIFT":
        return "left_shift"
    if key_name == "KEY_RIGHTCTRL":
        return "right_ctrl"
    if key_name == "KEY_LEFTCTRL":
        return "left_ctrl"
    if key_name == "KEY_RIGHTALT":
        return "right_alt"
    if key_name == "KEY_LEFTALT":
        return "left_alt"
    if key_name == "KEY_RIGHTMETA":
        return "right_cmd"
    if key_name == "KEY_LEFTMETA":
        return "left_cmd"
    if key_name == "KEY_SPACE":
        return "space"
    if key_name == "KEY_TAB":
        return "tab"
    if key_name == "KEY_ENTER":
        return "enter"
    if key_name == "KEY_ESC":
        return "esc"
    if key_name.startswith("KEY_F") and key_name[5:].isdigit():
        return key_name[4:].lower()
    if key_name.startswith("KEY_"):
        suffix = key_name[4:]
        if len(suffix) == 1 and suffix.isalpha():
            return suffix.lower()
        if suffix.isdigit():
            return suffix
        return f"key_{suffix.lower()}"
    return f"key_{key_name.lower()}"


def capture_hotkey_evdev(device):
    from evdev import ecodes

    for event in device.read_loop():
        if event.type != ecodes.EV_KEY or event.value != 1:
            continue
        try:
            key_name = ecodes.KEY[event.code]
        except Exception:
            key_name = None
        return hotkey_name_from_evdev(key_name)
    return None


def hotkey_name_from_pynput(key):
    from pynput import keyboard as pynput_keyboard

    if isinstance(key, pynput_keyboard.KeyCode) and key.char:
        return key.char.lower()

    special_map = {
        pynput_keyboard.Key.shift_r: "right_shift",
        pynput_keyboard.Key.shift_l: "left_shift",
        pynput_keyboard.Key.ctrl_r: "right_ctrl",
        pynput_keyboard.Key.ctrl_l: "left_ctrl",
        pynput_keyboard.Key.alt_r: "right_alt",
        pynput_keyboard.Key.alt_l: "left_alt",
        pynput_keyboard.Key.cmd_r: "right_cmd",
        pynput_keyboard.Key.cmd_l: "left_cmd",
        pynput_keyboard.Key.space: "space",
        pynput_keyboard.Key.tab: "tab",
        pynput_keyboard.Key.enter: "enter",
        pynput_keyboard.Key.esc: "esc",
    }
    mapped = special_map.get(key)
    if mapped:
        return mapped
    if hasattr(key, "name") and key.name:
        return key.name.lower()
    return None


def capture_hotkey_pynput():
    from pynput import keyboard as pynput_keyboard

    captured = {"key": None}
    done = threading.Event()

    def handle_press(key):
        captured["key"] = key
        done.set()
        return False

    with pynput_keyboard.Listener(on_press=handle_press) as listener:
        done.wait()
        listener.stop()
    return hotkey_name_from_pynput(captured["key"])


def select_evdev_device(devices, step_label=None):
    if len(devices) == 1 or not sys.stdin.isatty() or not config_prompts_enabled():
        return devices[0]
    label = f"{step_label} " if step_label else ""
    print(f"{label}Select keyboard device for hotkey:")
    for idx, device in enumerate(devices, start=1):
        print(f"  {idx}) {device.path} - {device.name}")
    while True:
        choice = input("> ").strip()
        if choice.isdigit():
            selection = int(choice)
            if 1 <= selection <= len(devices):
                return devices[selection - 1]
        print("Invalid selection. Enter a number from the list.")


def prompt_for_evdev_device(config_path, config, step_label=None):
    try:
        devices, permission_errors = list_evdev_keyboards()
    except Exception as exc:
        print(f"[hotkey] evdev backend unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    if not devices:
        print(
            "[hotkey] no accessible keyboard devices found for evdev.",
            file=sys.stderr,
        )
        if permission_errors:
            print(
                "[hotkey] try: sudo usermod -aG input $USER (then log out/in).",
                file=sys.stderr,
            )
        sys.exit(1)

    label = f"{step_label} " if step_label else ""
    saved_device = resolve_saved_evdev_device(config, devices)
    if saved_device is not None:
        current_text = f"{saved_device.name}"
        if not prompt_change_saved_value(
            label, "saved keyboard device", current_text
        ):
            save_config(config_path, config)
            for device in devices:
                if device is not saved_device:
                    device.close()
            return saved_device
    else:
        saved_name = str(config.get("evdev_device_name") or "").strip()
        saved_path = str(config.get("evdev_device_path") or "").strip()
        missing_value = saved_name or saved_path
        if missing_value:
            config.pop("evdev_device_path", None)
            print(
                f"{label}Saved keyboard device not available: {missing_value}"
            )
            save_config(config_path, config)

    selected = select_evdev_device(devices, step_label=step_label)
    config["evdev_device_name"] = selected.name
    config.pop("evdev_device_path", None)
    save_config(config_path, config)
    for device in devices:
        if device is not selected:
            device.close()
    return selected


def listen_hotkey_evdev(hotkey_code, on_press, on_release, selected_device=None):
    try:
        import evdev
        from evdev import ecodes
    except Exception as exc:
        print(f"[hotkey] evdev backend unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    selected = selected_device
    if selected is None:
        devices, permission_errors = find_evdev_devices(hotkey_code)
        if not devices:
            print(
                "[hotkey] no accessible keyboard devices found for evdev.",
                file=sys.stderr,
            )
            if permission_errors:
                print(
                    "[hotkey] try: sudo usermod -aG input $USER (then log out/in).",
                    file=sys.stderr,
                )
            sys.exit(1)

        selected = select_evdev_device(devices)
        for device in devices:
            if device is not selected:
                device.close()
    else:
        if not device_supports_hotkey(selected, hotkey_code):
            print(
                "[hotkey] selected device does not report the hotkey.",
                file=sys.stderr,
            )
            selected.close()
            sys.exit(1)

    print(
        f"[hotkey] listening with evdev on {selected.path} ({selected.name})"
    )
    try:
        for event in selected.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            if event.code != hotkey_code:
                continue
            if event.value == 1:
                on_press()
            elif event.value == 0:
                on_release()
    except KeyboardInterrupt:
        pass
    finally:
        selected.close()


def listen_hotkeys_evdev(hotkey_codes, on_press, on_release, selected_device=None):
    try:
        from evdev import ecodes
    except Exception as exc:
        print(f"[hotkey] evdev backend unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    codes = set(hotkey_codes)
    selected = selected_device
    if selected is None:
        devices, permission_errors = list_evdev_keyboards()
        matching = []
        for device in devices:
            if device_supports_any_hotkey(device, codes):
                matching.append(device)
            else:
                device.close()
        if not matching:
            print(
                "[hotkey] no accessible keyboard devices found for evdev.",
                file=sys.stderr,
            )
            if permission_errors:
                print(
                    "[hotkey] try: sudo usermod -aG input $USER (then log out/in).",
                    file=sys.stderr,
                )
            sys.exit(1)
        selected = select_evdev_device(matching)
        for device in matching:
            if device is not selected:
                device.close()
    elif not device_supports_any_hotkey(selected, codes):
        print(
            "[hotkey] selected device does not report the hotkey.",
            file=sys.stderr,
        )
        selected.close()
        sys.exit(1)

    print(
        f"[hotkey] listening with evdev on {selected.path} ({selected.name})"
    )
    try:
        for event in selected.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            if event.code not in codes:
                continue
            if event.value == 1:
                on_press()
            elif event.value == 0:
                on_release()
    except KeyboardInterrupt:
        pass
    finally:
        selected.close()


def listen_hotkey_pynput(hotkey, on_press, on_release):
    try:
        from pynput import keyboard as pynput_keyboard
    except Exception as exc:
        print(
            f"[hotkey] pynput backend unavailable: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    def handle_press(key):
        if key == hotkey:
            on_press()

    def handle_release(key):
        if key == hotkey:
            on_release()

    with pynput_keyboard.Listener(
        on_press=handle_press, on_release=handle_release
    ) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass


def listen_hotkeys_pynput(hotkeys, on_press, on_release):
    try:
        from pynput import keyboard as pynput_keyboard
    except Exception as exc:
        print(
            f"[hotkey] pynput backend unavailable: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    keys = set(hotkeys)

    def handle_press(key):
        if key in keys:
            on_press()

    def handle_release(key):
        if key in keys:
            on_release()

    with pynput_keyboard.Listener(
        on_press=handle_press, on_release=handle_release
    ) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass


def list_input_devices():
    if sd is None:
        raise RuntimeError(
            "sounddevice not installed; install it to list input devices."
        )
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = []
    for index, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            hostapi = hostapis[dev["hostapi"]]["name"]
            inputs.append(
                {
                    "index": index,
                    "name": dev["name"],
                    "hostapi": hostapi,
                    "channels": dev["max_input_channels"],
                    "rate": dev.get("default_samplerate", 0),
                }
            )
    return inputs


def prompt_for_device(config_path, config, step_label=None):
    if sd is None:
        raise RuntimeError(
            "sounddevice not installed; install it to select input devices."
        )

    devices = list_input_devices()
    if not devices:
        print("[config] no input devices found; using default.")
        return config.get("device")

    saved_device = resolve_saved_input_device(config, devices)
    if not sys.stdin.isatty() or not config_prompts_enabled():
        if saved_device is not None:
            save_config(config_path, config)
            return saved_device["index"]
        print("[config] prompts disabled; using default input device.")
        return config.get("device")

    default_input = (
        saved_device["index"] if saved_device is not None else config.get("device")
    )
    if default_input is not None and not any(
        dev["index"] == default_input for dev in devices
    ):
        default_input = None
    if default_input is None and sd.default.device:
        default_input = sd.default.device[0]

    label = f"{step_label} " if step_label else ""
    preserve_missing_saved_input = False
    if saved_device is not None:
        if not prompt_change_saved_value(
            label, "saved input device", format_input_device(saved_device)
        ):
            save_config(config_path, config)
            return saved_device["index"]
    else:
        saved_name = str(config.get("device_name") or "").strip()
        saved_hostapi = str(config.get("device_hostapi") or "").strip()
        missing_value = saved_name
        if saved_hostapi:
            missing_value = f"{saved_name} - {saved_hostapi}"
        elif config.get("device") is not None:
            missing_value = str(config.get("device"))
        if missing_value:
            print(f"{label}Saved input device not available: {missing_value}")
            preserve_missing_saved_input = has_saved_input_preference(config)

    print(f"{label}Select input device (press Enter for default):")
    for idx, dev in enumerate(devices, start=1):
        default_tag = ""
        if dev["index"] == default_input:
            default_tag = " (default)"
        print(f"  {idx}) {format_input_device(dev)}{default_tag}")

    while True:
        choice = input("> ").strip()
        if choice == "":
            default_device = next(
                (dev for dev in devices if dev["index"] == default_input), None
            )
            if default_device is not None:
                if preserve_missing_saved_input:
                    print(
                        f"{label}Using default input for this run; keeping "
                        "saved input device setting."
                    )
                    save_config(config_path, config)
                    return default_device["index"]
                save_input_device_selection(config, default_device)
            elif default_input is not None:
                if preserve_missing_saved_input:
                    print(
                        f"{label}Using saved input index for this run; keeping "
                        "saved input device setting."
                    )
                    save_config(config_path, config)
                    return default_input
                config["device"] = default_input
                config.pop("device_name", None)
                config.pop("device_hostapi", None)
            break
        if choice.isdigit():
            selection = int(choice)
            if 1 <= selection <= len(devices):
                save_input_device_selection(config, devices[selection - 1])
                break
        print("Invalid selection. Enter a number from the list.")

    save_config(config_path, config)
    return config.get("device")


def prompt_for_hotkey(config_path, config, backend, step_label=None, device=None):
    if not sys.stdin.isatty() or not config_prompts_enabled():
        print("[config] prompts disabled; using hotkey from config.")
        return normalize_hotkey_name(config.get("hotkey"))

    label = f"{step_label} " if step_label else ""
    saved_hotkey = normalize_hotkey_name(config.get("hotkey"))
    if saved_hotkey:
        if not prompt_change_saved_value(
            label, "saved hotkey", f"'{saved_hotkey}'"
        ):
            return saved_hotkey

    print(f"{label}Press the hotkey you want to hold to record:")

    while True:
        try:
            if backend == "evdev":
                if device is None:
                    print("[hotkey] evdev device not selected.", file=sys.stderr)
                    sys.exit(1)
                drain_evdev_events(device)
                wait_for_evdev_idle(device)
                hotkey_name = capture_hotkey_evdev(device)
                if not hotkey_name:
                    print("Unable to read a key. Try again.")
                    continue
                parse_hotkey_evdev(hotkey_name)
            else:
                hotkey_name = capture_hotkey_pynput()
                if not hotkey_name:
                    print("Unable to read a key. Try again.")
                    continue
                parse_hotkey_pynput(hotkey_name)
        except ValueError as exc:
            print(str(exc))
            continue
        except Exception as exc:
            backend_label = "evdev" if backend == "evdev" else "pynput"
            print(
                f"[hotkey] {backend_label} backend unavailable: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[hotkey] selected '{hotkey_name}'")
        config["hotkey"] = hotkey_name
        save_config(config_path, config)
        return hotkey_name


def write_wav(path, samples, sample_rate, channels):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def inspect_wav(path):
    try:
        with wave.open(path, "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except Exception as exc:
        return {"error": str(exc)}

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        return {
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_width": sample_width,
            "frames": frame_count,
            "duration": frame_count / float(sample_rate or 1),
            "bytes": len(raw),
            "rms": None,
            "peak": None,
        }

    data = np.frombuffer(raw, dtype=dtype)
    if channels > 1 and data.size:
        try:
            data = data.reshape(-1, channels)
        except ValueError:
            data = data.reshape(-1, 1)
    float_data = data.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(float_data ** 2))) if float_data.size else 0.0
    peak = float(np.max(np.abs(float_data))) if float_data.size else 0.0
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "frames": frame_count,
        "duration": frame_count / float(sample_rate or 1),
        "bytes": len(raw),
        "rms": rms,
        "peak": peak,
    }


def format_wav_stats(label, path, stats=None):
    stats = inspect_wav(path) if stats is None else stats
    if stats.get("error"):
        return f"{label}: path={path} error={stats['error']}"
    rms = stats.get("rms")
    peak = stats.get("peak")
    rms_text = "n/a" if rms is None else f"{rms:.2f}"
    peak_text = "n/a" if peak is None else f"{peak:.2f}"
    return (
        f"{label}: path={path} bytes={stats.get('bytes', 0)} "
        f"frames={stats.get('frames', 0)} duration={stats.get('duration', 0.0):.3f}s "
        f"rate={stats.get('sample_rate', 0)} channels={stats.get('channels', 0)} "
        f"width={stats.get('sample_width', 0)} rms={rms_text} peak={peak_text}"
    )


def transcribe_vosk(model, wav_path, sample_rate):
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(False)
    with wave.open(wav_path, "rb") as handle:
        while True:
            data = handle.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


def get_whisper_device(config):
    override = os.environ.get("VOICE_WHISPER_DEVICE") or config.get(
        "whisper_device"
    )
    if override:
        return str(override).strip()
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_whisper_model(config):
    override = os.environ.get("VOICE_WHISPER_MODEL")
    if override:
        return str(override).strip()
    return str(config.get("whisper_model", "base")).strip()


def requires_faster_whisper(model_name):
    value = str(model_name or "").strip().lower()
    return value.startswith("distil-") or value in ("large-v3-turbo",)


def get_whisper_fp16(config):
    override = os.environ.get("VOICE_WHISPER_FP16")
    if override is None:
        return config.get("whisper_fp16")
    value = str(override).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return config.get("whisper_fp16")


def describe_whisper_device(device):
    if device != "cuda":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cuda"
    if torch.version.hip:
        return f"cuda/rocm (hip {torch.version.hip})"
    if torch.version.cuda:
        return f"cuda ({torch.version.cuda})"
    return "cuda"


def log_whisper_gpu(device):
    if device != "cuda":
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown"
    if torch.version.cuda:
        runtime = f"cuda {torch.version.cuda}"
    elif torch.version.hip:
        runtime = f"hip {torch.version.hip}"
    else:
        runtime = "runtime unknown"
    print(f"[transcribe] gpu: {name} ({runtime})")


def load_whisper_model(model_name, device):
    try:
        import whisper
    except Exception as exc:
        print(
            f"[transcribe] whisper import failed: {exc}",
            file=sys.stderr,
        )
        print(
            "[transcribe] install: pip install openai-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    key = (model_name, device)
    if key not in WHISPER_MODEL_CACHE:
        WHISPER_MODEL_CACHE[key] = whisper.load_model(
            model_name, device=device
        )
    return WHISPER_MODEL_CACHE[key]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return None


def should_empty_torch_cache_after_transcribe(config, model):
    override = os.environ.get(
        "VOICE_TORCH_EMPTY_CACHE_AFTER_TRANSCRIBE"
    )
    if override is None:
        override = config.get("torch_empty_cache_after_transcribe")
    parsed_override = parse_bool(override)
    if parsed_override is not None:
        return parsed_override

    device = getattr(model, "device", None)
    if getattr(device, "type", None) != "cuda":
        return False

    try:
        import torch
    except Exception:
        return False
    return bool(getattr(torch.version, "hip", None))


def empty_torch_cache_after_transcribe(config, model):
    if not should_empty_torch_cache_after_transcribe(config, model):
        return
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if ipc_collect is not None:
                ipc_collect()
    except Exception as exc:
        print(
            f"[transcribe] torch cache cleanup failed: {exc}",
            file=sys.stderr,
        )


def transcribe_whisper(model, wav_path, config):
    options = {"verbose": False}
    language = config.get("whisper_language")
    if language:
        options["language"] = language
    task = config.get("whisper_task")
    if task:
        options["task"] = task
    fp16 = get_whisper_fp16(config)
    if fp16 is not None:
        use_fp16 = bool(fp16)
        if getattr(model, "device", None) is not None:
            if getattr(model.device, "type", "cpu") != "cuda":
                use_fp16 = False
        options["fp16"] = use_fp16
    result = None
    try:
        result = model.transcribe(wav_path, **options)
        return (result.get("text") or "").strip()
    finally:
        result = None
        empty_torch_cache_after_transcribe(config, model)


def get_faster_whisper_device(config):
    override = os.environ.get("VOICE_FASTER_WHISPER_DEVICE") or config.get(
        "faster_whisper_device"
    )
    if override:
        return str(override).strip()
    return "cpu"


def get_faster_whisper_compute_type(config):
    override = os.environ.get(
        "VOICE_FASTER_WHISPER_COMPUTE_TYPE"
    ) or config.get("faster_whisper_compute_type")
    if override:
        return str(override).strip()
    return "int8"


def get_remote_url(config):
    override = os.environ.get("VOICE_REMOTE_URL") or config.get("remote_url")
    if override:
        return str(override).strip()
    return "http://127.0.0.1:8765/transcribe"


def get_remote_timeout(config):
    override = os.environ.get("VOICE_REMOTE_TIMEOUT") or config.get(
        "remote_timeout"
    )
    if override is None:
        return 60.0
    try:
        return max(1.0, float(override))
    except (TypeError, ValueError):
        return 60.0


def get_transcribe_request_timeout(config):
    override = os.environ.get("VOICE_TRANSCRIBE_REQUEST_TIMEOUT") or config.get(
        "transcribe_request_timeout"
    )
    if override is None:
        return 120.0
    try:
        return max(0.0, float(override))
    except (TypeError, ValueError):
        return 120.0


def get_remote_wait_timeout(config):
    override = os.environ.get("VOICE_REMOTE_WAIT_TIMEOUT") or config.get(
        "remote_wait_timeout"
    )
    if override is None:
        return 300.0
    try:
        return max(1.0, float(override))
    except (TypeError, ValueError):
        return 300.0


def get_remote_wait_interval(config):
    override = os.environ.get("VOICE_REMOTE_WAIT_INTERVAL") or config.get(
        "remote_wait_interval"
    )
    if override is None:
        return 0.5
    try:
        return max(0.1, float(override))
    except (TypeError, ValueError):
        return 0.5


def get_remote_health_url(config):
    base_url = get_remote_url(config).rsplit("/", 1)[0]
    return f"{base_url}/health"


def is_remote_ready(config, timeout=5.0):
    import urllib.error
    import urllib.request

    health_url = get_remote_health_url(config)
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def wait_for_remote_ready(config):
    health_url = get_remote_health_url(config)
    timeout = get_remote_wait_timeout(config)
    interval = get_remote_wait_interval(config)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_remote_ready(config):
            return True
        time.sleep(interval)
    print(
        f"[transcribe] remote not ready after {timeout:.0f}s: {health_url}",
        file=sys.stderr,
    )
    return False


def resolve_initial_transcriber(config, sample_rate, channels):
    transcribe_audio, transcribe_backend, model_label = build_transcriber(
        config, sample_rate, channels
    )
    if transcribe_backend != "remote":
        return transcribe_audio, transcribe_backend, model_label

    remote_ready = wait_for_remote_ready(config)
    fallback_backend = get_fallback_backend(config)
    if (
        remote_ready
        or not fallback_backend
        or fallback_backend == transcribe_backend
    ):
        return transcribe_audio, transcribe_backend, model_label

    log_backend_fallback(
        transcribe_backend,
        fallback_backend,
        f"remote not ready: {get_remote_url(config)}",
    )
    return build_transcriber(
        config,
        sample_rate,
        channels,
        backend_override=fallback_backend,
        allow_fallback=False,
    )


def load_faster_whisper_model(model_name, device, compute_type):
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(
            f"[transcribe] faster-whisper import failed: {exc}",
            file=sys.stderr,
        )
        print(
            "[transcribe] install: pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    key = (model_name, device, compute_type)
    if key not in FASTER_WHISPER_MODEL_CACHE:
        FASTER_WHISPER_MODEL_CACHE[key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
    return FASTER_WHISPER_MODEL_CACHE[key]


def transcribe_faster_whisper(model, wav_path, config):
    task = config.get("whisper_task") or "transcribe"
    language = config.get("whisper_language")
    segments, _info = model.transcribe(
        wav_path,
        task=task,
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    parts = []
    for segment in segments:
        parts.append(segment.text)
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def get_nemo_device(config):
    override = os.environ.get("VOICE_NEMO_DEVICE") or config.get("nemo_device")
    if override:
        return str(override).strip()
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_nemo_model(config):
    override = os.environ.get("VOICE_NEMO_MODEL") or config.get("nemo_model")
    if override:
        return str(override).strip()
    return "nvidia/canary-qwen-2.5b"


def get_nemo_dtype(config):
    override = os.environ.get("VOICE_NEMO_DTYPE") or config.get(
        "nemo_dtype"
    )
    if override:
        value = str(override).strip().lower()
        if value in ("auto", ""):
            return (
                "bfloat16" if get_nemo_device(config) == "cuda" else "float32"
            )
        if value in ("none", "null"):
            return None
        return value
    return "bfloat16" if get_nemo_device(config) == "cuda" else "float32"


def get_nemo_max_new_tokens(config):
    value = os.environ.get("VOICE_NEMO_MAX_NEW_TOKENS")
    if value is None:
        value = config.get("nemo_max_new_tokens", 128)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 128


def use_nemo_subprocess_only(config):
    override = os.environ.get("VOICE_NEMO_SUBPROCESS_ONLY")
    if override is not None:
        return str(override).strip() == "1"
    return bool(config.get("nemo_subprocess_only"))


def get_nemo_engine(config):
    override = os.environ.get("VOICE_NEMO_ENGINE") or config.get(
        "nemo_engine"
    )
    if override:
        value = str(override).strip().lower()
        if value in ("salm", "speechlm2", "canary"):
            return "speechlm2"
        if value in ("asr", "parakeet"):
            return "asr"
        if value not in ("auto", ""):
            return value
    model_name = get_nemo_model(config).lower()
    if "canary" in model_name or "speechlm" in model_name:
        return "speechlm2"
    if "parakeet" in model_name or "tdt" in model_name:
        return "asr"
    return "asr"


def normalize_nemo_device(device):
    if device != "cuda":
        return device
    try:
        import torch
    except Exception:
        print(
            "[transcribe] torch not available; nemo falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    if not torch.cuda.is_available():
        print(
            "[transcribe] CUDA not available; nemo falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return device


def log_nemo_gpu(device):
    if device != "cuda":
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown"
    version = torch.version.cuda or "unknown"
    print(f"[transcribe] gpu: {name} (cuda {version})")


def resolve_local_hf_snapshot(model_name):
    if not model_name:
        return model_name
    if os.path.exists(model_name):
        return model_name
    return model_name


def patch_nemo_canary_lora():
    try:
        import nemo.collections.speechlm2.models.salm as salm_mod
    except Exception:
        return
    if getattr(salm_mod, "_voice_lora_patch", False):
        return
    original = salm_mod.maybe_install_lora

    def patched(model):
        llm = getattr(model, "llm", None)
        llm_model = getattr(llm, "model", None)
        embed_tokens = getattr(model, "embed_tokens", None)
        if (
            llm_model is not None
            and embed_tokens is not None
            and getattr(llm_model, "embed_tokens", None) is None
        ):
            llm_model.embed_tokens = embed_tokens
        return original(model)

    salm_mod.maybe_install_lora = patched
    salm_mod._voice_lora_patch = True


def resolve_local_canary_snapshot(model_name):
    if not model_name or os.path.exists(model_name) or "/" not in model_name:
        return model_name
    repo_dir = model_name.replace("/", "--")
    cache_roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/NeMo/NeMo_2.8.0rc0/hf_hub_cache"),
    ]
    for cache_root in cache_roots:
        snapshot_glob = os.path.join(
            cache_root, f"models--{repo_dir}", "snapshots", "*"
        )
        snapshot_dirs = sorted(
            (
                path
                for path in glob(snapshot_glob)
                if os.path.isdir(path)
            ),
            key=lambda path: os.path.getmtime(path),
            reverse=True,
        )
        for snapshot_dir in snapshot_dirs:
            if os.path.isfile(os.path.join(snapshot_dir, "config.json")) and (
                os.path.isfile(os.path.join(snapshot_dir, "model.safetensors"))
                or os.path.isfile(os.path.join(snapshot_dir, "pytorch_model.bin"))
            ):
                return snapshot_dir
    return model_name


def load_nemo_canary_model(
    model_name, device, dtype=None, force_reload=False
):
    try:
        from nemo.collections.speechlm2.models import SALM
    except Exception as exc:
        import traceback
        print(
            f"[transcribe] nemo import failed: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        print(
            '[transcribe] install: python -m pip install "nemo_toolkit[asr,tts] '
            '@ git+https://github.com/NVIDIA/NeMo.git"',
            file=sys.stderr,
        )
        sys.exit(1)

    patch_nemo_canary_lora()
    resolved_model_name = resolve_local_canary_snapshot(model_name)
    key = ("salm", resolved_model_name, device, dtype)
    if force_reload or key not in NEMO_MODEL_CACHE:
        model = SALM.from_pretrained(resolved_model_name)
        # Canary uses Triton CUDA kernels during generation; explicitly moving
        # the full model to CUDA has been observed to hang.
        use_explicit_cuda_placement = device != "cuda"
        try:
            import torch
        except Exception:
            torch = None
        if (
            torch is not None
            and isinstance(dtype, str)
            and use_explicit_cuda_placement
        ):
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is not None:
                try:
                    model = model.eval().to(torch_dtype)
                    if device:
                        model = model.to(device)
                except Exception:
                    if os.environ.get("VOICE_NEMO_DEBUG") == "1":
                        log_nemo_debug(
                            "failed to apply nemo dtype; using default"
                        )
                    model = model.eval()
        if device and use_explicit_cuda_placement:
            try:
                model.to(device)
            except Exception:
                pass
        try:
            model.eval()
        except Exception:
            pass
        if os.environ.get("VOICE_NEMO_DEBUG") == "1":
            try:
                log_nemo_debug(
                    f"loaded dtype={next(model.parameters()).dtype}"
                )
            except Exception:
                pass
        NEMO_MODEL_CACHE[key] = model
    return NEMO_MODEL_CACHE[key]


def load_nemo_asr_model(model_name, device, dtype=None, force_reload=False):
    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as exc:
        import traceback
        print(
            f"[transcribe] nemo asr import failed: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        print(
            '[transcribe] install: python -m pip install "nemo_toolkit[asr,tts] '
            '@ git+https://github.com/NVIDIA/NeMo.git"',
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_model_name = resolve_local_hf_snapshot(model_name)
    key = ("asr", resolved_model_name, device, dtype)
    if force_reload or key not in NEMO_MODEL_CACHE:
        model = ASRModel.from_pretrained(resolved_model_name)
        try:
            import torch
        except Exception:
            torch = None
        if device:
            try:
                model = model.to(device)
            except Exception:
                pass
        if torch is not None and isinstance(dtype, str) and device == "cuda":
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is not None:
                try:
                    model = model.to(torch_dtype)
                except Exception:
                    pass
        try:
            model.eval()
        except Exception:
            pass
        NEMO_MODEL_CACHE[key] = model
    return NEMO_MODEL_CACHE[key]


def ensure_nemo_wav(wav_path, target_rate=16000):
    try:
        with wave.open(wav_path, "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            sampwidth = handle.getsampwidth()
    except Exception:
        return wav_path, None

    if channels == 1 and rate == target_rate and sampwidth == 2:
        return wav_path, None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-sample_fmt",
            "s16",
            tmp_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            return tmp_path, tmp_path
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            print(
                f"[transcribe] ffmpeg convert failed: {detail}",
                file=sys.stderr,
            )
    else:
        try:
            import numpy as np
            import soundfile as sf  # type: ignore

            data, rate = sf.read(wav_path, always_2d=True)
            data = data.astype("float32", copy=False)
            if data.shape[1] > 1:
                data = np.mean(data, axis=1)
            else:
                data = data[:, 0]
            if rate != target_rate:
                try:
                    import librosa  # type: ignore

                    data = librosa.resample(
                        data, orig_sr=rate, target_sr=target_rate
                    )
                except Exception:
                    try:
                        from scipy.signal import resample_poly  # type: ignore

                        data = resample_poly(data, target_rate, rate)
                    except Exception:
                        data = None
                rate = target_rate
            if data is not None:
                sf.write(tmp_path, data, rate, subtype="PCM_16")
                return tmp_path, tmp_path
        except Exception as exc:
            print(
                f"[transcribe] python audio convert failed: {exc}",
                file=sys.stderr,
            )
        print(
            "[transcribe] ffmpeg not found; nemo expects 16k mono s16 audio.",
            file=sys.stderr,
        )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    return wav_path, None


def load_nemo_audio_tensor(wav_path, target_rate=16000, device=None):
    try:
        import torch
    except Exception:
        return None, None

    audio = None
    rate = None
    try:
        import torchaudio  # type: ignore

        audio, rate = torchaudio.load(wav_path)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if rate != target_rate:
            audio = torchaudio.functional.resample(audio, rate, target_rate)
    except Exception:
        try:
            import soundfile as sf  # type: ignore
            import numpy as np

            data, rate = sf.read(wav_path, always_2d=True)
            data = data.astype("float32", copy=False)
            if data.shape[1] > 1:
                data = np.mean(data, axis=1, keepdims=True)
            data = data.T  # shape (1, time)
            if rate != target_rate:
                try:
                    import librosa  # type: ignore

                    data = librosa.resample(
                        data.squeeze(0), orig_sr=rate, target_sr=target_rate
                    ).reshape(1, -1)
                except Exception:
                    try:
                        from scipy.signal import resample_poly  # type: ignore

                        data = resample_poly(
                            data.squeeze(0), target_rate, rate
                        ).reshape(1, -1)
                    except Exception:
                        return None, None
                rate = target_rate
            audio = torch.from_numpy(data)
        except Exception:
            return None, None

    if audio is None or rate is None:
        return None, None
    if os.environ.get("VOICE_NEMO_DEBUG") == "1":
        try:
            rms = float(torch.sqrt(torch.mean(audio ** 2)))
            peak = float(torch.max(torch.abs(audio)))
            log_nemo_debug(
                f"audio shape={tuple(audio.shape)} rate={rate} rms={rms:.6f} peak={peak:.6f}"
            )
        except Exception:
            pass
    if device is not None:
        audio = audio.to(device)
    audio_lens = torch.tensor([audio.shape[-1]], device=audio.device)
    return audio, audio_lens


def normalize_nemo_text(text):
    if text is None:
        return ""
    cleaned = str(text)
    for token in (
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|endoftext|>",
    ):
        cleaned = cleaned.replace(token, "")
    return sanitize_transcript_text(cleaned)


def normalize_transcript_for_filter(text):
    cleaned = "".join(
        ch.lower() if ch.isalnum() or ch.isspace() else " "
        for ch in str(text or "")
    )
    return " ".join(cleaned.split())


def has_repetitive_transcript_pattern(normalized_text):
    tokens = normalized_text.split()
    if len(tokens) < 6:
        return False

    max_run = 1
    run = 1
    for prev_token, token in zip(tokens, tokens[1:]):
        if token == prev_token:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run >= 3:
        return True

    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(tokens) >= 0.45:
        return True

    short_token_count = sum(1 for token in tokens if len(token) <= 2)
    if short_token_count / len(tokens) >= 0.6 and most_common_count >= 3:
        return True

    bigrams = list(zip(tokens, tokens[1:]))
    if len(bigrams) >= 4:
        repeated_bigram_count = Counter(bigrams).most_common(1)[0][1]
        if (
            repeated_bigram_count >= 3
            and repeated_bigram_count / len(bigrams) >= 0.3
        ):
            return True

    return False


def is_likely_bad_transcript(text):
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    normalized = normalize_transcript_for_filter(stripped)
    if normalized in {
        "we ll see you next time",
        "see you next time",
        "thanks for watching",
        "thank you for watching",
    }:
        return True
    if has_repetitive_transcript_pattern(normalized):
        return True
    alnum = sum(ch.isalnum() for ch in stripped)
    if alnum == 0:
        return True
    unique = {ch for ch in stripped}
    if len(stripped) >= 12 and len(unique) <= 2:
        return True
    return False


def decode_nemo_output(model, answer_ids):
    def score(text):
        if not text:
            return (-1, 0)
        alnum = sum(ch.isalnum() for ch in text)
        return (alnum, len(text))

    ids = answer_ids
    if hasattr(answer_ids, "dim"):
        try:
            if answer_ids.dim() >= 2:
                ids = answer_ids[0]
        except Exception:
            pass
    elif isinstance(answer_ids, (list, tuple)) and answer_ids:
        if not isinstance(answer_ids[0], (int, np.integer)):
            ids = answer_ids[0]
    if hasattr(ids, "cpu"):
        try:
            ids = ids.cpu()
        except Exception:
            pass
    try:
        ids_list = ids.tolist()
    except Exception:
        ids_list = ids

    tokenizer = getattr(model, "tokenizer", None)
    candidates = []
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            text = tokenizer.decode(ids_list, skip_special_tokens=True)
            candidates.append(normalize_nemo_text(text))
        except Exception:
            pass
    if tokenizer is not None and hasattr(tokenizer, "ids_to_text"):
        try:
            text = tokenizer.ids_to_text(ids)
            candidates.append(normalize_nemo_text(text))
        except Exception:
            pass
    if not candidates:
        return normalize_nemo_text(ids_list)
    return max(candidates, key=score)


def parse_nemo_hyp(answer, eos_tokens):
    if not eos_tokens:
        return answer
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(answer, torch.Tensor):
        end = torch.isin(
            answer, torch.tensor(eos_tokens, device=answer.device)
        ).nonzero(as_tuple=True)[0]
        if end.numel() == 0:
            return answer
        return answer[: end[0]]
    try:
        for idx, token in enumerate(answer):
            if token in eos_tokens:
                return answer[:idx]
    except Exception:
        pass
    return answer


def normalize_nemo_transcribe_result(result):
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        result = result[0]
    if isinstance(result, dict):
        for key in ("text", "predicted_text", "transcription"):
            if key in result:
                return normalize_nemo_text(result[key])
        return normalize_nemo_text(result)
    return normalize_nemo_text(result)


def normalize_nemo_asr_result(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return normalize_nemo_text(result)
    if isinstance(result, dict):
        for key in ("text", "pred_text", "predicted_text", "transcription"):
            if key in result:
                return normalize_nemo_text(result[key])
        return normalize_nemo_text(result)
    text = getattr(result, "text", None)
    if text:
        return normalize_nemo_text(text)
    return normalize_nemo_text(result)


def transcribe_nemo_asr(model, wav_path, config):
    wav_path, cleanup_path = ensure_nemo_wav(wav_path, target_rate=16000)
    try:
        use_lhotse = config.get("nemo_use_lhotse")
        if use_lhotse is None:
            use_lhotse = False
        result = model.transcribe(
            [wav_path],
            batch_size=1,
            num_workers=0,
            use_lhotse=bool(use_lhotse),
        )
        if isinstance(result, (list, tuple)) and result:
            return normalize_nemo_asr_result(result[0])
        return normalize_nemo_asr_result(result)
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


def transcribe_nemo_canary(model, wav_path, config):
    max_new_tokens = get_nemo_max_new_tokens(config)
    audio_tag = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    user_prompt = os.environ.get("VOICE_NEMO_USER_PROMPT")
    if not user_prompt:
        user_prompt = config.get("nemo_user_prompt")
    system_prompt = os.environ.get("VOICE_NEMO_SYSTEM_PROMPT")
    if not system_prompt:
        system_prompt = config.get("nemo_system_prompt")
    wav_path, cleanup_path = ensure_nemo_wav(wav_path, target_rate=16000)
    try:
        turns = []
        if system_prompt:
            turns.append(
                {"role": "system", "content": str(system_prompt)}
            )
        content = str(
            user_prompt
            or "These are spoken coding commands for an AI agent. Transcribe them exactly:"
        )
        content = f"{content} {audio_tag}"
        turns.append(
            {
                "role": "user",
                "content": content,
                "audio": [wav_path],
            }
        )
        if os.environ.get("VOICE_NEMO_DEBUG") == "1":
            log_nemo_debug(
                f"prompt content={content!r} audio={wav_path!r}"
            )

        prompts = [turns]

        def run_generate(active_model):
            answer_ids = active_model.generate(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
            if hasattr(answer_ids, "cpu"):
                answer_ids = answer_ids.cpu()
            if os.environ.get("VOICE_NEMO_DEBUG") == "1":
                log_nemo_debug(
                    f"answer_ids type={type(answer_ids)} shape={getattr(answer_ids, 'shape', None)}"
                )
            eos_tokens = [getattr(active_model, "text_eos_id", None)]
            eos_tokens = [tok for tok in eos_tokens if tok is not None]
            try:
                if eos_tokens:
                    if hasattr(answer_ids, "dim") and answer_ids.dim() >= 2:
                        parsed = parse_nemo_hyp(
                            answer_ids[0], eos_tokens
                        )
                    elif (
                        isinstance(answer_ids, (list, tuple))
                        and answer_ids
                        and isinstance(answer_ids[0], (list, tuple))
                    ):
                        parsed = parse_nemo_hyp(
                            answer_ids[0], eos_tokens
                        )
                    else:
                        parsed = parse_nemo_hyp(answer_ids, eos_tokens)
                else:
                    parsed = answer_ids
            except Exception:
                parsed = answer_ids
            text = decode_nemo_output(active_model, parsed)
            log_nemo_debug(f"decoded='{text}'")
            return text

        try:
            text = run_generate(model)
            if text and not is_likely_bad_transcript(text):
                return text
        except Exception as exc:
            log_nemo_debug(f"generate failed: {exc}")
            if is_nemo_cuda_failure(exc):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    cpu_model = load_nemo_canary_model(
                        get_nemo_model(config),
                        "cpu",
                        "float32",
                        force_reload=True,
                    )
                    text = run_generate(cpu_model)
                    if text and not is_likely_bad_transcript(text):
                        return text
                except Exception as cpu_exc:
                    log_nemo_debug(f"cpu fallback failed: {cpu_exc}")
            text = ""
        if os.environ.get("VOICE_NEMO_RETRY", "1") != "0":
            try:
                retry_model = load_nemo_canary_model(
                    get_nemo_model(config),
                    normalize_nemo_device(get_nemo_device(config)),
                    get_nemo_dtype(config),
                    force_reload=True,
                )
                text = run_generate(retry_model)
                if text and not is_likely_bad_transcript(text):
                    return text
            except Exception as exc:
                log_nemo_debug(f"retry failed: {exc}")
        if os.environ.get("VOICE_NEMO_SUBPROCESS", "1") != "0":
            try:
                text = transcribe_nemo_subprocess(wav_path, config)
                if text and not is_likely_bad_transcript(text):
                    return text
            except Exception as exc:
                log_nemo_debug(f"subprocess failed: {exc}")
        return ""
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


def transcribe_nemo_subprocess(wav_path, config):
    script_path = os.environ.get(
        "VOICE_NEMO_SUBPROCESS_SCRIPT", "/app/nemo_subprocess.py"
    )
    timeout = config.get("nemo_subprocess_timeout", 300)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 300
    cmd = ["python3", script_path, wav_path]
    env = os.environ.copy()
    env["VOICE_NEMO_MODEL"] = get_nemo_model(config)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    for line in reversed(output.splitlines()):
        if line.startswith("__TRANSCRIPT__:"):
            return line.split(":", 1)[1].strip()
    return ""


class RemoteTranscribeError(RuntimeError):
    pass


def transcribe_remote(wav_path, config):
    import urllib.error
    import urllib.request

    url = get_remote_url(config)
    timeout = get_remote_timeout(config)
    request_timeout = get_transcribe_request_timeout(config)
    if request_timeout > 0:
        timeout = min(timeout, request_timeout)
    raise_on_error = bool(get_fallback_backend(config))
    try:
        if not is_remote_ready(config):
            message = f"remote not ready: {get_remote_health_url(config)}"
            print(f"[transcribe] {message}", file=sys.stderr)
            if raise_on_error:
                raise RemoteTranscribeError(message)
            return ""
        stats = inspect_wav(wav_path)
        print(
            f"[transcribe] {format_wav_stats('client wav', wav_path, stats)}",
            file=sys.stderr,
        )
        with open(wav_path, "rb") as handle:
            data = handle.read()
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "audio/wav"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = getattr(resp, "status", 200)
        if status != 200:
            detail = body.decode("utf-8", errors="ignore").strip()
            message = f"remote error: {status} body={detail}"
            print(f"[transcribe] {message}", file=sys.stderr)
            if raise_on_error:
                raise RemoteTranscribeError(message)
            return ""
        try:
            payload = json.loads(body.decode("utf-8"))
            return (payload.get("text") or "").strip()
        except Exception:
            return body.decode("utf-8", errors="ignore").strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        message = f"remote http error: {exc.code} body={detail}"
        print(f"[transcribe] {message}", file=sys.stderr)
        if raise_on_error:
            raise RemoteTranscribeError(message) from exc
        return ""
    except urllib.error.URLError as exc:
        message = f"remote request failed: {exc}"
        print(f"[transcribe] {message}", file=sys.stderr)
        if raise_on_error:
            raise RemoteTranscribeError(message) from exc
        return ""
    except Exception as exc:
        message = f"remote error: {exc}"
        print(f"[transcribe] {message}", file=sys.stderr)
        if raise_on_error:
            raise RemoteTranscribeError(message) from exc
        return ""


def get_sherpa_model_dir(config):
    return os.environ.get(
        "VOICE_SHERPA_MODEL_DIR",
        config.get("sherpa_model_dir", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
    )


def get_sherpa_provider(config):
    return os.environ.get(
        "VOICE_SHERPA_PROVIDER",
        config.get("sherpa_provider", "cpu"),
    )


def get_sherpa_model_type(config):
    value = os.environ.get(
        "VOICE_SHERPA_MODEL_TYPE",
        config.get("sherpa_model_type", "auto"),
    )
    model_type = str(value).strip().lower()
    if model_type not in ("auto", "transducer", "whisper"):
        print(
            f"[transcribe] invalid sherpa_model_type '{value}'. "
            "Use auto, transducer, or whisper.",
            file=sys.stderr,
        )
        sys.exit(1)
    return model_type


def _pick_first_existing(path_candidates):
    for path in path_candidates:
        if os.path.isfile(path):
            return path
    return None


def _pick_first_glob(patterns):
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_sherpa_model_paths(model_dir, configured_type):
    transducer = {
        "encoder": _pick_first_existing(
            [
                f"{model_dir}/encoder.int8.onnx",
                f"{model_dir}/encoder.onnx",
            ]
        ),
        "decoder": _pick_first_existing(
            [
                f"{model_dir}/decoder.int8.onnx",
                f"{model_dir}/decoder.onnx",
            ]
        ),
        "joiner": _pick_first_existing(
            [
                f"{model_dir}/joiner.int8.onnx",
                f"{model_dir}/joiner.onnx",
            ]
        ),
        "tokens": f"{model_dir}/tokens.txt",
    }
    has_transducer = (
        transducer["encoder"] is not None
        and transducer["decoder"] is not None
        and transducer["joiner"] is not None
        and os.path.isfile(transducer["tokens"])
    )

    whisper = {
        "encoder": _pick_first_glob(
            [
                f"{model_dir}/*encoder*.onnx",
                f"{model_dir}/encoder*.onnx",
            ]
        ),
        "decoder": _pick_first_glob(
            [
                f"{model_dir}/*decoder*.onnx",
                f"{model_dir}/decoder*.onnx",
            ]
        ),
        "tokens": _pick_first_glob(
            [
                f"{model_dir}/*tokens*.txt",
                f"{model_dir}/tokens*.txt",
            ]
        ),
    }
    has_whisper = (
        whisper["encoder"] is not None
        and whisper["decoder"] is not None
        and whisper["tokens"] is not None
    )

    model_type = configured_type
    if model_type == "auto":
        if has_transducer:
            model_type = "transducer"
        elif has_whisper:
            model_type = "whisper"

    if model_type == "transducer" and has_transducer:
        return model_type, transducer
    if model_type == "whisper" and has_whisper:
        return model_type, whisper

    print(
        f"[transcribe] sherpa model files not found for type='{configured_type}' at '{model_dir}'.",
        file=sys.stderr,
    )
    print(
        "[transcribe] expected transducer files: encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt",
        file=sys.stderr,
    )
    print(
        "[transcribe] expected whisper files: *encoder*.onnx, *decoder*.onnx, *tokens*.txt",
        file=sys.stderr,
    )
    print(
        "[transcribe] tiny.en download guide: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html",
        file=sys.stderr,
    )
    sys.exit(1)


def load_sherpa_model(model_dir, provider, configured_type="auto"):
    if not os.path.isdir(model_dir):
        print(
            f"[transcribe] sherpa model not found at '{model_dir}'.",
            file=sys.stderr,
        )
        print(
            "[transcribe] Download with:",
            file=sys.stderr,
        )
        print(
            "  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
            file=sys.stderr,
        )
        print(
            "  tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
            file=sys.stderr,
        )
        sys.exit(1)

    model_type, model_paths = resolve_sherpa_model_paths(
        model_dir, configured_type
    )
    cache_key = (model_dir, provider, model_type)
    if cache_key in SHERPA_MODEL_CACHE:
        return SHERPA_MODEL_CACHE[cache_key], model_type

    try:
        import sherpa_onnx
    except ImportError:
        print(
            "[transcribe] sherpa-onnx not installed. Install with:",
            file=sys.stderr,
        )
        print(
            "  pip install sherpa-onnx",
            file=sys.stderr,
        )
        sys.exit(1)

    if model_type == "whisper":
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=model_paths["encoder"],
            decoder=model_paths["decoder"],
            tokens=model_paths["tokens"],
            provider=provider,
        )
    else:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model_paths["encoder"],
            decoder=model_paths["decoder"],
            joiner=model_paths["joiner"],
            tokens=model_paths["tokens"],
            model_type="nemo_transducer",
            provider=provider,
        )
    SHERPA_MODEL_CACHE[cache_key] = recognizer
    return recognizer, model_type


def transcribe_sherpa(recognizer, wav_path):
    import soundfile as sf

    samples, sample_rate = sf.read(wav_path, dtype="float32")
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def get_parakeet_onnx_model(config):
    return os.environ.get(
        "VOICE_PARAKEET_ONNX_MODEL",
        config.get("parakeet_onnx_model", "istupakov/parakeet-tdt-0.6b-v3-onnx"),
    )


def get_parakeet_onnx_quantization(config):
    return os.environ.get(
        "VOICE_PARAKEET_ONNX_QUANTIZATION",
        config.get("parakeet_onnx_quantization", "int8"),
    )


def resolve_parakeet_onnx_model_alias(model_name):
    if model_name in (
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
    ):
        return "nemo-parakeet-tdt-0.6b-v3"
    return model_name


def get_huggingface_cache_dir():
    try:
        from huggingface_hub import constants as hf_constants

        return (
            os.path.expanduser(str(hf_constants.HF_HUB_CACHE)),
            "huggingface_hub",
        )
    except Exception:
        pass

    cache_dir = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    if cache_dir:
        return os.path.expanduser(cache_dir), "env"

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(os.path.expanduser(hf_home), "hub"), "HF_HOME"

    return os.path.expanduser("~/.cache/huggingface/hub"), "default"


def log_parakeet_onnx_download_info(model_name, resolved_model, quantization):
    cache_dir, source = get_huggingface_cache_dir()
    print(
        "[transcribe] parakeet-onnx loading "
        f"requested='{model_name}' resolved='{resolved_model}' "
        f"quantization='{quantization}'"
    )
    print(
        f"[transcribe] parakeet-onnx download/cache dir: {cache_dir} "
        f"({source})"
    )
    disabled = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "").strip().lower()
    if disabled in ("1", "true", "yes", "on"):
        print(
            "[transcribe] Hugging Face download progress bars are disabled by "
            "HF_HUB_DISABLE_PROGRESS_BARS.",
            file=sys.stderr,
        )
    else:
        print(
            "[transcribe] download progress, if needed, is shown by onnx-asr/"
            "Hugging Face Hub in this output."
        )


def load_parakeet_onnx_model(model_name, quantization):
    resolved_model = resolve_parakeet_onnx_model_alias(model_name)
    cache_key = (resolved_model, quantization)
    if cache_key in PARAKEET_ONNX_MODEL_CACHE:
        return PARAKEET_ONNX_MODEL_CACHE[cache_key]

    try:
        import onnx_asr
    except ImportError:
        print(
            "[transcribe] onnx-asr not installed. Install with:",
            file=sys.stderr,
        )
        print(
            "  python -m pip install 'onnx-asr[cpu,hub]'",
            file=sys.stderr,
        )
        sys.exit(1)

    if resolved_model != model_name:
        print(
            f"[transcribe] parakeet-onnx alias '{resolved_model}' for '{model_name}'"
        )

    log_parakeet_onnx_download_info(model_name, resolved_model, quantization)
    model = onnx_asr.load_model(resolved_model, quantization=quantization)
    PARAKEET_ONNX_MODEL_CACHE[cache_key] = model
    return model


def transcribe_parakeet_onnx(model, wav_path):
    import soundfile as sf

    normalized_path = wav_path
    cleanup_path = None
    try:
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        if samples.shape[1] > 1:
            samples = np.mean(samples, axis=1, dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                sf.write(handle.name, samples, sample_rate)
                normalized_path = handle.name
                cleanup_path = normalized_path

        result = model.recognize(normalized_path)
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result).strip()
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


def get_transcript_correction_backend(config):
    value = os.environ.get("VOICE_TRANSCRIPT_CORRECTION_BACKEND")
    if value is None:
        value = config.get("transcript_correction_backend", "off")
    normalized = str(value or "").strip().lower()
    if normalized in ("", "0", "false", "no", "none", "null", "off"):
        return "off"
    if normalized in (
        "gemma",
        "gemma4",
        "gemma-4",
        "llama",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ):
        return "llama-cpp"
    return normalized


def get_transcript_correction_max_new_tokens(config):
    return get_config_int(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_MAX_NEW_TOKENS",
        "transcript_correction_max_new_tokens",
        96,
        minimum=8,
    )


def get_transcript_correction_max_chars(config):
    return get_config_int(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_MAX_CHARS",
        "transcript_correction_max_chars",
        700,
        minimum=40,
    )


def get_transcript_correction_llama_cpp_path(config):
    return get_config_string(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_PATH",
        "transcript_correction_llama_cpp_path",
        "llama-cli",
    )


def get_transcript_correction_llama_cpp_server_path(config):
    configured = get_config_string(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_SERVER_PATH",
        "transcript_correction_llama_cpp_server_path",
        "",
    )
    if configured:
        return configured
    binary_path = get_transcript_correction_llama_cpp_path(config)
    if binary_path:
        return os.path.join(os.path.dirname(binary_path), "llama-server")
    return "llama-server"


def get_transcript_correction_llama_cpp_server_url(config):
    return get_config_string(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_SERVER_URL",
        "transcript_correction_llama_cpp_server_url",
        "http://127.0.0.1:18087",
    ).rstrip("/")


def transcript_correction_llama_cpp_server_autostart(config):
    value = os.environ.get(
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_SERVER_AUTOSTART"
    )
    if value is None:
        value = config.get(
            "transcript_correction_llama_cpp_server_autostart",
            True,
        )
    parsed = parse_bool(value)
    return bool(parsed) if parsed is not None else True


def get_transcript_correction_llama_cpp_server_startup_timeout(config):
    return get_config_float(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_SERVER_STARTUP_TIMEOUT",
        "transcript_correction_llama_cpp_server_startup_timeout",
        60.0,
        minimum=0.5,
    )


def get_transcript_correction_llama_cpp_model(config):
    return get_config_string(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL",
        "transcript_correction_llama_cpp_model",
        DEFAULT_CONFIG["transcript_correction_llama_cpp_model"],
    )


def get_transcript_correction_llama_cpp_gpu_layers(config):
    return get_config_int(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_GPU_LAYERS",
        "transcript_correction_llama_cpp_gpu_layers",
        99,
        minimum=0,
    )


def get_transcript_correction_llama_cpp_timeout(config):
    return get_config_float(
        config,
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_TIMEOUT",
        "transcript_correction_llama_cpp_timeout",
        20.0,
        minimum=0.5,
    )


def transcript_correction_applies_to_probes(config):
    value = os.environ.get("VOICE_TRANSCRIPT_CORRECTION_APPLY_TO_PROBES")
    if value is None:
        value = config.get("transcript_correction_apply_to_probes", False)
    parsed = parse_bool(value)
    return bool(parsed) if parsed is not None else False


def transcript_correction_console_log_enabled(config):
    value = os.environ.get("VOICE_TRANSCRIPT_CORRECTION_CONSOLE_LOG")
    if value is None:
        value = config.get("transcript_correction_console_log", True)
    parsed = parse_bool(value)
    return bool(parsed) if parsed is not None else True


def build_transcript_correction_messages(text, command_labels, config):
    configured_prompt = os.environ.get("VOICE_TRANSCRIPT_CORRECTION_PROMPT")
    if configured_prompt is None:
        configured_prompt = config.get("transcript_correction_prompt")
    system_prompt = str(configured_prompt or "")
    labels = sorted({str(label).strip() for label in command_labels or [] if label})
    target_text = ", ".join(labels[:24]) if labels else "none"
    user_prompt = (
        f"Available spoken routing targets: {target_text}\n"
        f"Raw transcript: {text}\n"
        "Corrected transcript:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_terminal_control_chars(text):
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text or ""))
    while "\b" in cleaned:
        updated = re.sub(r".\x08", "", cleaned)
        if updated == cleaned:
            cleaned = cleaned.replace("\b", "")
            break
        cleaned = updated
    return "".join(
        ch for ch in cleaned if ch in ("\n", "\t") or ord(ch) >= 32
    )


def parse_llama_cpp_response(output_text):
    cleaned = strip_terminal_control_chars(output_text)
    candidates = []
    ignored_prefixes = (
        "Loading model",
        "build",
        "model",
        "modalities",
        "using custom system prompt",
        "available commands",
        "/exit",
        "/regen",
        "/clear",
        "/read",
        "/glob",
        "[ Prompt:",
        "Exiting",
    )
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("|").strip()
        if not line or line.startswith(">"):
            continue
        if any(line.startswith(prefix) for prefix in ignored_prefixes):
            continue
        if set(line) <= {"▄", "█", "▀", " "}:
            continue
        candidates.append(line)
    return candidates[-1] if candidates else ""


def resolve_local_runtime_path(path):
    value = os.path.expanduser(str(path or "").strip())
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    if os.sep not in value:
        found = shutil.which(value)
        if found:
            return found
    root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(root, value)
    if os.path.exists(candidate):
        return candidate
    return value


def add_binary_library_path(env, binary_path):
    binary_dir = os.path.dirname(os.path.abspath(binary_path))
    if not binary_dir:
        return
    existing_library_path = env.get("LD_LIBRARY_PATH")
    if existing_library_path:
        env["LD_LIBRARY_PATH"] = binary_dir + os.pathsep + existing_library_path
    else:
        env["LD_LIBRARY_PATH"] = binary_dir


def llama_cpp_server_models_url(config):
    return get_transcript_correction_llama_cpp_server_url(config) + "/v1/models"


def llama_cpp_server_ready(config):
    url = llama_cpp_server_models_url(config)
    try:
        with urllib.request.urlopen(
            url,
            timeout=min(2.0, get_transcript_correction_llama_cpp_timeout(config)),
        ) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except Exception:
        return False


def wait_for_llama_cpp_server(config, process=None):
    deadline = time.monotonic() + get_transcript_correction_llama_cpp_server_startup_timeout(config)
    while time.monotonic() < deadline:
        if llama_cpp_server_ready(config):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.25)
    return llama_cpp_server_ready(config)


def ensure_llama_cpp_server(config):
    global LLAMA_CPP_SERVER_PROCESS

    if llama_cpp_server_ready(config):
        return
    if LLAMA_CPP_SERVER_PROCESS is not None:
        if LLAMA_CPP_SERVER_PROCESS.poll() is None:
            if wait_for_llama_cpp_server(config, LLAMA_CPP_SERVER_PROCESS):
                return
        LLAMA_CPP_SERVER_PROCESS = None

    if not transcript_correction_llama_cpp_server_autostart(config):
        raise RuntimeError("llama.cpp server is not running")

    server_path = resolve_local_runtime_path(
        get_transcript_correction_llama_cpp_server_path(config)
    )
    model_path = resolve_local_runtime_path(
        get_transcript_correction_llama_cpp_model(config)
    )
    if not server_path or not os.path.isfile(server_path):
        raise RuntimeError(f"llama.cpp server binary not found: {server_path}")
    if not os.access(server_path, os.X_OK):
        raise RuntimeError(f"llama.cpp server binary is not executable: {server_path}")
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError(f"llama.cpp transcript correction model not found: {model_path}")

    parsed = urllib.parse.urlparse(
        get_transcript_correction_llama_cpp_server_url(config)
    )
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18087
    argv = [
        server_path,
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        str(get_transcript_correction_llama_cpp_gpu_layers(config)),
        "--ctx-size",
        "2048",
        "-n",
        str(get_transcript_correction_max_new_tokens(config)),
        "--jinja",
        "--reasoning",
        "off",
        "--log-disable",
        "--no-webui",
    ]
    env = os.environ.copy()
    add_binary_library_path(env, server_path)
    log_path = os.path.join(
        os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir(),
        "speech-agent-workbench-llama-server.log",
    )
    log_handle = open(log_path, "a", encoding="utf-8")
    try:
        LLAMA_CPP_SERVER_PROCESS = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        log_handle.close()
    if not wait_for_llama_cpp_server(config, LLAMA_CPP_SERVER_PROCESS):
        raise RuntimeError(
            f"llama.cpp server did not become ready; see {log_path}"
        )
    print(
        "[transcribe] llama.cpp server ready: "
        + get_transcript_correction_llama_cpp_server_url(config)
    )


def correct_transcript_with_llama_cpp_server(text, config, command_labels=None):
    ensure_llama_cpp_server(config)
    messages = build_transcript_correction_messages(text, command_labels, config)
    payload = {
        "messages": messages,
        "max_tokens": get_transcript_correction_max_new_tokens(config),
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        get_transcript_correction_llama_cpp_server_url(config)
        + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=get_transcript_correction_llama_cpp_timeout(config),
    ) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def run_llama_cpp_chat_completion(config, messages, max_tokens=None, timeout=None):
    ensure_llama_cpp_server(config)
    payload = {
        "messages": messages,
        "max_tokens": int(max_tokens or get_transcript_correction_max_new_tokens(config)),
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        get_transcript_correction_llama_cpp_server_url(config)
        + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout or get_transcript_correction_llama_cpp_timeout(config),
    ) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def correct_transcript_with_llama_cpp(text, config, command_labels=None):
    binary_path = resolve_local_runtime_path(
        get_transcript_correction_llama_cpp_path(config)
    )
    model_path = resolve_local_runtime_path(
        get_transcript_correction_llama_cpp_model(config)
    )
    if not binary_path:
        raise RuntimeError("llama.cpp transcript correction path is not configured")
    if not model_path:
        raise RuntimeError("llama.cpp transcript correction model is not configured")

    messages = build_transcript_correction_messages(text, command_labels, config)
    argv = [
        binary_path,
        "-m",
        model_path,
        "-cnv",
        "-st",
        "--jinja",
        "--reasoning",
        "off",
        "-sys",
        messages[0]["content"],
        "-p",
        messages[1]["content"],
        "-n",
        str(get_transcript_correction_max_new_tokens(config)),
        "-ngl",
        str(get_transcript_correction_llama_cpp_gpu_layers(config)),
        "--temp",
        "0",
        "--seed",
        "1",
        "--no-display-prompt",
        "--no-warmup",
        "--no-show-timings",
    ]
    env = os.environ.copy()
    add_binary_library_path(env, binary_path)
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=get_transcript_correction_llama_cpp_timeout(config),
        env=env,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        tail = "\n".join(strip_terminal_control_chars(output).splitlines()[-8:])
        raise RuntimeError(
            f"llama.cpp transcript correction failed with exit "
            f"{completed.returncode}: {tail}"
        )
    return parse_llama_cpp_response(output)


def corrected_transcript_is_plausible(raw_text, corrected_text):
    corrected = str(corrected_text or "").strip()
    if not corrected:
        return False
    if is_likely_bad_transcript(corrected):
        return False
    raw = str(raw_text or "").strip()
    if not raw:
        return False
    if len(corrected) > max(80, len(raw) * 3):
        return False
    raw_words = set(normalize_transcript_for_filter(raw).split())
    corrected_words = set(normalize_transcript_for_filter(corrected).split())
    if (
        len(raw_words) >= 3
        and corrected_words
        and raw_words.isdisjoint(corrected_words)
    ):
        return False
    return True


def make_transcript_correction_details(
    raw_text,
    pre_llm_text,
    corrected_text,
    backend,
    model_output=None,
    model_accepted=None,
    model_skipped=False,
    fallback_reason=None,
):
    return {
        "raw_transcript": compact_history_text(raw_text),
        "pre_llm_transcript": compact_history_text(pre_llm_text),
        "corrected_transcript": compact_history_text(corrected_text),
        "correction_backend": backend,
        "model_output": compact_history_text(model_output)
        if model_output is not None
        else None,
        "model_accepted": model_accepted,
        "model_skipped": bool(model_skipped),
        "fallback_reason": fallback_reason,
    }


def truncate_transcript_log_text(value, limit=500):
    text = compact_history_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def log_transcript_correction_details(config, details):
    if not transcript_correction_console_log_enabled(config):
        return
    if not details or details.get("correction_backend") != "llama-cpp":
        return
    print(
        "[transcribe] transcript correction raw: "
        + truncate_transcript_log_text(details.get("raw_transcript"))
    )
    print(
        "[transcribe] transcript correction pre-llm: "
        + truncate_transcript_log_text(details.get("pre_llm_transcript"))
    )
    model_output = details.get("model_output")
    if details.get("model_skipped"):
        reason = details.get("fallback_reason") or "skipped"
        print(f"[transcribe] transcript correction llama.cpp skipped: {reason}")
    elif model_output is not None:
        status = "accepted" if details.get("model_accepted") else "rejected"
        print(
            f"[transcribe] transcript correction llama.cpp {status}: "
            + truncate_transcript_log_text(model_output)
        )
    print(
        "[transcribe] transcript correction final: "
        + truncate_transcript_log_text(details.get("corrected_transcript"))
    )


def finalize_transcript_correction_details(config, details):
    log_transcript_correction_details(config, details)
    return details


def combine_transcript_correction_details(chunk_details, final_details):
    chunks = [detail for detail in chunk_details or [] if detail]
    if not chunks:
        return final_details
    combined = dict(final_details or {})
    combined["raw_transcript"] = " ".join(
        compact_history_text(detail.get("raw_transcript"))
        for detail in chunks
        if detail.get("raw_transcript")
    ).strip()
    combined["pre_llm_transcript"] = " ".join(
        compact_history_text(detail.get("pre_llm_transcript"))
        for detail in chunks
        if detail.get("pre_llm_transcript")
    ).strip()
    return combined


def correct_transcript_details(text, config, command_labels=None, skip_model=False):
    raw_text = str(text or "")
    corrected = correct_common_coding_terms(raw_text)
    backend = get_transcript_correction_backend(config)
    if backend == "off" or skip_model:
        final_text = sanitize_transcript_text(corrected)
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_skipped=True,
            ),
        )
    if backend != "llama-cpp":
        print(
            f"[transcribe] unknown transcript correction backend '{backend}'; "
            "using uncorrected transcript.",
            file=sys.stderr,
        )
        final_text = sanitize_transcript_text(corrected)
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_skipped=True,
                fallback_reason="unknown_backend",
            ),
        )
    max_chars = get_transcript_correction_max_chars(config)
    if len(corrected) > max_chars:
        final_text = sanitize_transcript_text(corrected)
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_skipped=True,
                fallback_reason="max_chars",
            ),
        )

    failure_key = (backend, get_transcript_correction_llama_cpp_model(config))
    if failure_key in TRANSCRIPT_CORRECTION_FAILURES:
        final_text = sanitize_transcript_text(corrected)
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_skipped=True,
                fallback_reason="previous_failure",
            ),
        )

    try:
        with LLAMA_CPP_CORRECTOR_LOCK:
            try:
                model_text = correct_transcript_with_llama_cpp_server(
                    corrected,
                    config,
                    command_labels=command_labels,
                )
            except Exception as server_exc:
                if transcript_correction_llama_cpp_server_autostart(config):
                    print(
                        "[transcribe] llama.cpp server unavailable; "
                        f"falling back to one-shot CLI: {server_exc}",
                        file=sys.stderr,
                    )
                model_text = correct_transcript_with_llama_cpp(
                    corrected,
                    config,
                    command_labels=command_labels,
                )
    except Exception as exc:
        TRANSCRIPT_CORRECTION_FAILURES.add(failure_key)
        print(
            f"[transcribe] transcript correction disabled after error: {exc}",
            file=sys.stderr,
        )
        final_text = sanitize_transcript_text(corrected)
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_output=None,
                model_accepted=False,
                model_skipped=True,
                fallback_reason="model_error",
            ),
        )

    if corrected_transcript_is_plausible(corrected, model_text):
        final_text = sanitize_transcript_text(correct_common_coding_terms(model_text))
        return finalize_transcript_correction_details(
            config,
            make_transcript_correction_details(
                raw_text,
                corrected,
                final_text,
                backend,
                model_output=model_text,
                model_accepted=True,
            ),
        )
    final_text = sanitize_transcript_text(corrected)
    return finalize_transcript_correction_details(
        config,
        make_transcript_correction_details(
            raw_text,
            corrected,
            final_text,
            backend,
            model_output=model_text,
            model_accepted=False,
            fallback_reason="implausible_model_output",
        ),
    )


def correct_transcript_text(text, config, command_labels=None, skip_model=False):
    return correct_transcript_details(
        text,
        config,
        command_labels=command_labels,
        skip_model=skip_model,
    )["corrected_transcript"]


def start_transcript_correction_server_background(config):
    if get_transcript_correction_backend(config) != "llama-cpp":
        return
    if not transcript_correction_llama_cpp_server_autostart(config):
        return

    def worker():
        try:
            with LLAMA_CPP_CORRECTOR_LOCK:
                ensure_llama_cpp_server(config)
        except Exception as exc:
            print(
                f"[transcribe] llama.cpp server background start failed: {exc}",
                file=sys.stderr,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def build_transcriber(
    config,
    sample_rate,
    channels,
    backend_override=None,
    allow_fallback=True,
):
    if backend_override is None:
        backend_override = os.environ.get("VOICE_TRANSCRIBE_BACKEND")
    transcribe_backend = normalize_backend(
        backend_override or config.get("transcribe_backend")
    )
    fallback_backend = (
        get_fallback_backend(config) if allow_fallback else None
    )
    model_path = config.get("model_path")
    model_label = "unknown"

    try:
        if transcribe_backend == "vosk":
            if Model is None or KaldiRecognizer is None:
                print(
                    "[transcribe] vosk not available; install with pip install vosk.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not os.path.isdir(model_path):
                print(
                    f"Missing Vosk model at '{model_path}'.\n"
                    "Download a model and set model_path in config.json.",
                    file=sys.stderr,
                )
                sys.exit(1)
            model = Model(model_path)
            transcribe_audio = lambda wav_path: transcribe_vosk(
                model, wav_path, sample_rate
            )
            model_label = str(model_path)
        elif transcribe_backend == "whisper":
            if not shutil.which("ffmpeg"):
                print(
                    "[transcribe] ffmpeg not found; whisper requires ffmpeg.",
                    file=sys.stderr,
                )
                print(
                    "[transcribe] install: sudo apt-get install -y ffmpeg",
                    file=sys.stderr,
                )
            whisper_device = get_whisper_device(config)
            model_name = get_whisper_model(config)
            if requires_faster_whisper(model_name):
                print(
                    f"[transcribe] model '{model_name}' is routed to faster-whisper.",
                )
                transcribe_backend = "faster-whisper"
                fw_device = get_faster_whisper_device(config)
                fw_compute = get_faster_whisper_compute_type(config)
                model = load_faster_whisper_model(
                    model_name, fw_device, fw_compute
                )
                transcribe_audio = lambda wav_path: transcribe_faster_whisper(
                    model, wav_path, config
                )
                model_label = (
                    f"{model_name} on {fw_device} ({fw_compute})"
                )
                print(
                    f"[transcribe] faster-whisper model '{model_name}' on "
                    f"{fw_device} ({fw_compute})"
                )
            else:
                model = load_whisper_model(model_name, whisper_device)
                transcribe_audio = lambda wav_path: transcribe_whisper(
                    model, wav_path, config
                )
                model_label = (
                    f"{model_name} on {describe_whisper_device(whisper_device)}"
                )
                print(
                    f"[transcribe] whisper model '{model_name}' on "
                    f"{describe_whisper_device(whisper_device)}"
                )
                log_whisper_gpu(whisper_device)
                if whisper_device != "cuda":
                    print(
                        "[transcribe] GPU not available; using CPU.",
                        file=sys.stderr,
                    )
        elif transcribe_backend == "faster-whisper":
            if not shutil.which("ffmpeg"):
                print(
                    "[transcribe] ffmpeg not found; faster-whisper requires ffmpeg.",
                    file=sys.stderr,
                )
                print(
                    "[transcribe] install: sudo apt-get install -y ffmpeg",
                    file=sys.stderr,
                )
            fw_device = get_faster_whisper_device(config)
            fw_compute = get_faster_whisper_compute_type(config)
            model_name = get_whisper_model(config)
            model = load_faster_whisper_model(model_name, fw_device, fw_compute)
            transcribe_audio = lambda wav_path: transcribe_faster_whisper(
                model, wav_path, config
            )
            model_label = f"{model_name} on {fw_device} ({fw_compute})"
            print(
                f"[transcribe] faster-whisper model '{model_name}' on "
                f"{fw_device} ({fw_compute})"
            )
        elif transcribe_backend in ("nemo", "canary", "nemo-canary"):
            if sample_rate != 16000 or channels != 1:
                print(
                    "[transcribe] nemo expects 16000 Hz mono audio; "
                    "check sample_rate/channels.",
                    file=sys.stderr,
                )
            model_name = get_nemo_model(config)
            nemo_engine = get_nemo_engine(config)
            if use_nemo_subprocess_only(config):
                transcribe_audio = lambda wav_path: transcribe_nemo_subprocess(
                    wav_path, config
                )
                model_label = f"{model_name} via subprocess"
                print(
                    f"[transcribe] nemo subprocess model '{model_name}'"
                )
            else:
                nemo_device = get_nemo_device(config)
                nemo_device = normalize_nemo_device(nemo_device)
                nemo_dtype = get_nemo_dtype(config)

                def transcribe_audio(wav_path):
                    engine = (
                        get_nemo_engine(config)
                        if nemo_engine == "auto"
                        else nemo_engine
                    )
                    if engine == "speechlm2":
                        model = load_nemo_canary_model(
                            model_name, nemo_device, nemo_dtype
                        )
                        return transcribe_nemo_canary(
                            model, wav_path, config
                        )
                    model = load_nemo_asr_model(
                        model_name, nemo_device, nemo_dtype
                    )
                    return transcribe_nemo_asr(model, wav_path, config)

                model_label = (
                    f"{model_name} on {nemo_device} (engine={nemo_engine})"
                )
                print(
                    f"[transcribe] nemo {nemo_engine} model '{model_name}' on {nemo_device}"
                )
                log_nemo_gpu(nemo_device)
                if nemo_device != "cuda":
                    print(
                        "[transcribe] GPU not available; using CPU.",
                        file=sys.stderr,
                    )
        elif transcribe_backend == "remote":
            transcribe_audio = lambda wav_path: transcribe_remote(
                wav_path, config
            )
            model_label = get_remote_url(config)
            print(
                f"[transcribe] remote backend -> {get_remote_url(config)}"
            )
        elif transcribe_backend == "sherpa":
            model_dir = get_sherpa_model_dir(config)
            provider = get_sherpa_provider(config)
            configured_model_type = get_sherpa_model_type(config)
            recognizer, resolved_model_type = load_sherpa_model(
                model_dir, provider, configured_model_type
            )
            transcribe_audio = lambda wav_path: transcribe_sherpa(
                recognizer, wav_path
            )
            model_label = (
                f"{model_dir} ({provider}, type={resolved_model_type})"
            )
            print(
                f"[transcribe] sherpa-onnx model '{model_dir}' "
                f"({provider}, type={resolved_model_type})"
            )
        elif transcribe_backend in ("parakeet", "parakeet-onnx"):
            model_name = get_parakeet_onnx_model(config)
            quantization = get_parakeet_onnx_quantization(config)
            model = load_parakeet_onnx_model(model_name, quantization)
            transcribe_audio = lambda wav_path: transcribe_parakeet_onnx(
                model, wav_path
            )
            model_label = f"{model_name} (quantization={quantization})"
            print(
                f"[transcribe] parakeet-onnx model '{model_name}' "
                f"(quantization={quantization})"
            )
        else:
            print(
                f"[transcribe] unknown backend '{transcribe_backend}'.",
                file=sys.stderr,
            )
            sys.exit(1)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        if fallback_backend and fallback_backend != transcribe_backend:
            log_backend_fallback(
                transcribe_backend, fallback_backend, f"startup error: {exc}"
            )
            return build_transcriber(
                config,
                sample_rate,
                channels,
                backend_override=fallback_backend,
                allow_fallback=False,
            )
        raise

    backend_transcribe_audio = transcribe_audio

    def transcribe_audio(wav_path):
        return sanitize_transcript_text(backend_transcribe_audio(wav_path))

    if fallback_backend and fallback_backend != transcribe_backend:
        fallback_state = {}
        primary_transcribe_audio = transcribe_audio
        primary_model_label = model_label

        def get_fallback_transcriber():
            if "value" not in fallback_state:
                fallback_state["value"] = build_transcriber(
                    config,
                    sample_rate,
                    channels,
                    backend_override=fallback_backend,
                    allow_fallback=False,
                )
            return fallback_state["value"]

        def transcribe_audio(wav_path):
            try:
                text = primary_transcribe_audio(wav_path)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                log_backend_fallback(
                    transcribe_backend,
                    fallback_backend,
                    f"request error: {exc}",
                )
                fallback_audio, _fb_backend, _fb_model = (
                    get_fallback_transcriber()
                )
                return fallback_audio(wav_path)
            if text and is_likely_bad_transcript(text):
                log_backend_fallback(
                    transcribe_backend,
                    fallback_backend,
                    f"suspicious transcript: {text!r}",
                )
                fallback_audio, _fb_backend, _fb_model = (
                    get_fallback_transcriber()
                )
                return fallback_audio(wav_path)
            return text

        model_label = f"{primary_model_label} -> fallback {fallback_backend}"

    return transcribe_audio, transcribe_backend, model_label


def run_command(argv, input_text=None, timeout=None):
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def get_env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def get_env_float(name, default, minimum=0.0):
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return max(float(minimum), parsed)


def paste_debug_enabled():
    value = os.environ.get("VOICE_PASTE_DEBUG", "")
    if value.strip():
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(PASTE_DEBUG_DEFAULT)


def log_paste_debug(message):
    if paste_debug_enabled():
        print(f"[paste] {message}")


def get_auto_focus_log_path():
    value = os.environ.get("VOICE_AUTO_FOCUS_LOG", "").strip()
    if not value or value.lower() in ("0", "false", "no", "off"):
        return None
    return os.path.expanduser(value)


def summarize_log_text(value, limit=400):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def append_auto_focus_log(event, **fields):
    path = get_auto_focus_log_path()
    if not path:
        return False
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
    }
    record.update(fields)
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        return True
    except OSError as exc:
        print(f"[auto] unable to write focus log {path}: {exc}", file=sys.stderr)
        return False


def get_auto_tmux_console_log_path(config):
    value = os.environ.get("VOICE_AUTO_TMUX_CONSOLE_LOG")
    if value is None:
        value = config.get(
            "auto_tmux_console_log",
            DEFAULT_CONFIG["auto_tmux_console_log"],
        )
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in ("0", "false", "no", "none", "null", "off"):
        return None
    return os.path.expanduser(value)


def split_tmux_console_log_record(line):
    text = str(line or "").rstrip("\n").replace("\x00", "")
    if text.startswith("{"):
        try:
            record = json.loads(text)
            timestamp = record.get("ts")
            if timestamp is not None:
                timestamp = float(timestamp)
            return timestamp, str(record.get("data") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    timestamp_text, separator, payload = text.partition("\t")
    if separator and timestamp_text.isdigit():
        try:
            return float(timestamp_text), payload
        except ValueError:
            pass
    return None, text


def trim_auto_tmux_console_log(path, retention_seconds, max_bytes):
    if not path:
        return 0
    retention_seconds = max(0.0, float(retention_seconds or 0.0))
    max_bytes = max(0, int(max_bytes or 0))
    if retention_seconds <= 0 and max_bytes <= 0:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    try:
        with open(path, "r+", encoding="utf-8", errors="replace") as handle:
            data = handle.read()
            lines = data.splitlines(keepends=True)
            if retention_seconds > 0:
                cutoff = time.time() - retention_seconds
                kept_lines = []
                for line in lines:
                    timestamp, _payload = split_tmux_console_log_record(line)
                    if timestamp is None or timestamp >= cutoff:
                        kept_lines.append(line)
            else:
                kept_lines = lines

            trimmed = "".join(kept_lines)
            if max_bytes > 0:
                encoded = trimmed.encode("utf-8", errors="replace")
                if len(encoded) > max_bytes:
                    tail = encoded[-max_bytes:]
                    first_newline = tail.find(b"\n")
                    if first_newline >= 0 and first_newline + 1 < len(tail):
                        tail = tail[first_newline + 1 :]
                    trimmed = tail.decode("utf-8", errors="replace")

            if trimmed != data:
                handle.seek(0)
                handle.write(trimmed)
                handle.truncate()
                handle.flush()
            return len(trimmed.encode("utf-8", errors="replace"))
    except FileNotFoundError:
        return 0
    except OSError as exc:
        print(f"[auto] unable to trim tmux console log {path}: {exc}", file=sys.stderr)
        try:
            return os.path.getsize(path)
        except OSError:
            return 0


def tmux_console_buffer_should_flush(buffer, last_update_at, idle_seconds, now=None):
    if not buffer or last_update_at is None:
        return False
    if idle_seconds <= 0:
        return True
    current = time.monotonic() if now is None else now
    return current - last_update_at >= idle_seconds


def flush_tmux_console_buffer(buffer):
    if not buffer:
        return False
    print("".join(buffer), end="", flush=True)
    buffer.clear()
    return True


def extract_tmux_console_agent_payload(payload):
    text = str(payload or "")
    match = re.search(r"\[tmux\]\[([^\]]+)\]\s*", text)
    if not match:
        return None, text
    label = match.group(1).strip()
    cleaned = re.sub(
        r"(^|[\n\r])\[tmux\]\[[^\]]+\]\s*",
        lambda item: item.group(1),
        text,
    )
    return label, cleaned


def tmux_console_payload_to_lines(payload):
    cleaned = strip_terminal_control_chars(str(payload or "").replace("\r", "\n"))
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def trim_tmux_agent_lines(lines, max_lines):
    max_lines = max(1, int(max_lines or 1))
    if len(lines) > max_lines:
        del lines[: len(lines) - max_lines]
    return lines


def tmux_summary_digest(command_text, lines):
    payload = json.dumps(
        {
            "command": str(command_text or ""),
            "lines": list(lines or []),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def normalize_tmux_agent_key(value):
    return normalize_voice_command_text(value)


def record_tmux_sent_command(label, target, text):
    record = {
        "label": str(label or "").strip(),
        "target": str(target or "").strip(),
        "text": str(text or "").strip(),
        "ts": time.time(),
    }
    for key in (
        normalize_tmux_agent_key(record["label"]),
        record["target"],
    ):
        if key:
            TMUX_RECENT_COMMANDS[key] = record
    return record


def get_tmux_sent_command(label=None, target=None):
    for key in (
        normalize_tmux_agent_key(label),
        str(target or "").strip(),
    ):
        if key and key in TMUX_RECENT_COMMANDS:
            return TMUX_RECENT_COMMANDS[key]
    return None


def build_tmux_summary_messages(agent_label, command_text, lines):
    command = str(command_text or "").strip() or "unknown"
    output = "\n".join(str(line) for line in lines or [])
    return [
        {
            "role": "system",
            "content": (
                "You summarize terminal output for a voice-controlled coding "
                "workbench. Return exactly one concise sentence. Say whether "
                "the task appears complete, failed, blocked, or still in progress "
                "when the output supports it. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent: {agent_label or 'unknown'}\n"
                f"Original command: {command}\n"
                "Last terminal output lines:\n"
                f"{output}\n\n"
                "One-sentence summary:"
            ),
        },
    ]


def clean_one_sentence_summary(text):
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1).strip() if match else cleaned


def summarize_tmux_agent_output(config, agent_label, command_text, lines):
    max_tokens = get_config_int(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_MAX_NEW_TOKENS",
        "auto_tmux_summary_max_new_tokens",
        DEFAULT_CONFIG["auto_tmux_summary_max_new_tokens"],
        minimum=16,
    )
    messages = build_tmux_summary_messages(agent_label, command_text, lines)
    summary = run_llama_cpp_chat_completion(
        config,
        messages,
        max_tokens=max_tokens,
        timeout=get_transcript_correction_llama_cpp_timeout(config),
    )
    return clean_one_sentence_summary(summary)


def print_tmux_summary(agent_label, command_text, summary):
    label = agent_label or "tmux"
    command = str(command_text or "").strip() or "unknown"
    summary = str(summary or "").strip() or "No summary was returned."
    print(f"\n[tmux-summary][{label}] Command: {command}")
    print(f"[tmux-summary][{label}] Summary: {summary}", flush=True)


SENSITIVE_URL_QUERY_KEYS = {
    "access_token",
    "api_key",
    "auth",
    "authorization",
    "key",
    "secret",
    "signature",
    "sig",
    "token",
}


def redact_url_for_log(url):
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return text
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"***@{netloc.rsplit('@', 1)[1]}"
    query = parts.query
    if query:
        query_items = urllib.parse.parse_qsl(query, keep_blank_values=True)
        redacted_items = []
        for key, value in query_items:
            if key.lower() in SENSITIVE_URL_QUERY_KEYS:
                redacted_items.append((key, "***"))
            else:
                redacted_items.append((key, value))
        query = urllib.parse.urlencode(redacted_items, safe="*")
    return urllib.parse.urlunsplit(
        (parts.scheme, netloc, parts.path, query, parts.fragment)
    )


def get_tmux_summary_webhook_url(config):
    return get_config_string(
        config,
        "VOICE_TMUX_SUMMARY_WEBHOOK_URL",
        "tmux_summary_webhook_url",
        DEFAULT_CONFIG["tmux_summary_webhook_url"],
    )


def get_tmux_summary_webhook_token(config):
    return get_config_string(
        config,
        "VOICE_TMUX_SUMMARY_WEBHOOK_TOKEN",
        "tmux_summary_webhook_token",
        DEFAULT_CONFIG["tmux_summary_webhook_token"],
    )


def get_tmux_summary_webhook_timeout(config):
    return get_config_float(
        config,
        "VOICE_TMUX_SUMMARY_WEBHOOK_TIMEOUT",
        "tmux_summary_webhook_timeout",
        DEFAULT_CONFIG["tmux_summary_webhook_timeout"],
        minimum=0.1,
    )


def log_tmux_summary_webhook_configuration(config):
    url = get_tmux_summary_webhook_url(config)
    summary_enabled = get_config_bool(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_ENABLED",
        "auto_tmux_summary_enabled",
        DEFAULT_CONFIG["auto_tmux_summary_enabled"],
    )
    if not url:
        print(
            "[tmux-summary] webhook disabled; set "
            "VOICE_TMUX_SUMMARY_WEBHOOK_URL to POST summaries",
            flush=True,
        )
        return
    token = get_tmux_summary_webhook_token(config)
    timeout = get_tmux_summary_webhook_timeout(config)
    auth_status = "bearer token configured" if token else "not configured"
    print(f"[tmux-summary] webhook POST {redact_url_for_log(url)}", flush=True)
    print(
        f"[tmux-summary] webhook auth: {auth_status}; timeout: {timeout:g}s",
        flush=True,
    )
    if not summary_enabled:
        print(
            "[tmux-summary] webhook inactive because tmux summaries are disabled",
            flush=True,
        )


def post_tmux_summary_webhook(config, agent_label, command_text, summary):
    url = get_tmux_summary_webhook_url(config)
    if not url:
        return False
    payload = {
        "agent": str(agent_label or ""),
        "command": str(command_text or ""),
        "summary": str(summary or ""),
        "timestamp": time.time(),
    }
    headers = {"Content-Type": "application/json"}
    token = get_tmux_summary_webhook_token(config)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = get_tmux_summary_webhook_timeout(config)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except Exception as exc:
        print(f"[tmux-summary] webhook failed: {exc}", file=sys.stderr)
        return False


def dispatch_tmux_summary_webhook(config, agent_label, command_text, summary):
    if not get_tmux_summary_webhook_url(config):
        return None
    thread = threading.Thread(
        target=post_tmux_summary_webhook,
        args=(config, agent_label, command_text, summary),
        daemon=True,
    )
    thread.start()
    return thread


def parse_agent_prefixed_message(text):
    message = str(text or "").strip()
    match = re.match(r"^([^:\n]{1,80}):\s*(.+)$", message, flags=re.S)
    if not match:
        return None, message
    agent = match.group(1).strip()
    body = match.group(2).strip()
    if not agent or not body:
        return None, message
    return agent, body


def parse_api_message_payload(body, content_type="application/json"):
    content_type = str(content_type or "").split(";", 1)[0].strip().lower()
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body or "")
    if content_type == "application/json" or text.lstrip().startswith("{"):
        try:
            payload = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            return None, None, f"invalid_json: {exc}"
        if not isinstance(payload, dict):
            return None, None, "invalid_payload"
        agent = str(payload.get("agent") or "").strip()
        message = str(payload.get("message") or payload.get("text") or "").strip()
        if agent and message:
            return agent, message, None
        if message:
            agent, body_text = parse_agent_prefixed_message(message)
            if agent and body_text:
                return agent, body_text, None
        return None, None, "missing_message"

    agent, body_text = parse_agent_prefixed_message(text)
    if not agent or not body_text:
        return None, None, "missing_agent_prefix"
    return agent, body_text, None


def build_api_agent_command_index(commands):
    index = {}
    labels = {}
    for alias, command in (commands or {}).items():
        if not command.get("tmux_send_target"):
            continue
        label = str(command.get("label") or alias or "").strip()
        if label:
            labels.setdefault(normalize_voice_command_text(label), label)
        candidates = {str(alias or ""), label}
        normalized_label = normalize_voice_command_text(label)
        if normalized_label:
            candidates.update(build_command_text_aliases(normalized_label))
        for candidate in candidates:
            key = normalize_voice_command_text(candidate)
            if key:
                index.setdefault(key, command)
    available = sorted(labels.values(), key=lambda value: value.lower())
    return index, available


def route_api_message_to_tmux(agent, message, commands):
    index, available = build_api_agent_command_index(commands)
    key = normalize_voice_command_text(agent)
    command = index.get(key)
    if command is None:
        print(
            f"[api] rejected message for unknown agent '{str(agent or '').strip()}'; "
            f"available: {', '.join(available) if available else 'none'}",
            flush=True,
        )
        return {
            "ok": False,
            "error": "unknown_agent",
            "agent": str(agent or ""),
            "available_agents": available,
        }
    body = str(message or "").strip()
    if not body:
        print(
            f"[api] rejected empty message for {command.get('label') or agent}",
            flush=True,
        )
        return {
            "ok": False,
            "error": "empty_message",
            "agent": command.get("label") or agent,
            "available_agents": available,
        }
    focused = run_auto_shell_command(command)
    if not focused:
        print(
            f"[api] failed to focus {command.get('label') or agent}; "
            "message was not sent",
            flush=True,
        )
        return {
            "ok": False,
            "error": "focus_failed",
            "agent": command.get("label") or agent,
            "message": body,
            "focused": False,
            "sent": False,
        }
    sent = send_text_to_tmux_target(command, body)
    status = "routed" if sent else "failed to route"
    print(
        f"[api] {status} message to {command.get('label') or agent} "
        f"({len(body)} chars)",
        flush=True,
    )
    return {
        "ok": bool(sent),
        "agent": command.get("label") or agent,
        "focused": bool(focused),
        "message": body,
        "sent": bool(sent),
    }


def voice_api_authorized(headers, token):
    token = str(token or "").strip()
    if not token:
        return True
    auth_header = headers.get("Authorization", "")
    if auth_header == f"Bearer {token}":
        return True
    return headers.get("X-Voice-Api-Token", "") == token


def get_voice_api_bind(config):
    host = get_config_string(
        config,
        "VOICE_API_HOST",
        "api_host",
        DEFAULT_CONFIG["api_host"],
    ) or "127.0.0.1"
    port = get_config_int(
        config,
        "VOICE_API_PORT",
        "api_port",
        DEFAULT_CONFIG["api_port"],
        minimum=1,
    )
    return host, port


def get_voice_api_token(config):
    return get_config_string(
        config,
        "VOICE_API_TOKEN",
        "api_token",
        DEFAULT_CONFIG["api_token"],
    )


def format_voice_api_url_host(host):
    host = str(host or "127.0.0.1").strip() or "127.0.0.1"
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host


def build_voice_api_post_url(host, port):
    return f"http://{format_voice_api_url_host(host)}:{int(port)}/messages"


def log_voice_api_configuration(config, commands, enabled):
    host, port = get_voice_api_bind(config)
    post_url = build_voice_api_post_url(host, port)
    token = get_voice_api_token(config)
    _index, available = build_api_agent_command_index(commands)
    if not enabled:
        print(
            f"[api] local message API disabled; enable VOICE_API_ENABLED=1 "
            f"for POST {post_url}",
            flush=True,
        )
        return
    print(f"[api] POST {post_url}", flush=True)
    if format_voice_api_url_host(host) != host:
        print(f"[api] bound to {host}:{port}", flush=True)
    print(
        "[api] payload: "
        '{"message":"flux: pull the latest"} or '
        '{"agent":"Flux","message":"pull the latest"}',
        flush=True,
    )
    auth_status = (
        "bearer token required via Authorization or X-Voice-Api-Token"
        if token
        else "not configured"
    )
    print(f"[api] auth: {auth_status}", flush=True)
    print(
        f"[api] agents: {', '.join(available) if available else 'none configured'}",
        flush=True,
    )


def make_voice_api_handler(config, commands):
    token = get_voice_api_token(config)

    class VoiceApiHandler(http.server.BaseHTTPRequestHandler):
        server_version = "SpeechAgentWorkbenchAPI/1.0"

        def log_message(self, _format, *_args):
            return

        def send_json(self, status_code, payload):
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
            if path != "/messages":
                self.send_json(404, {"ok": False, "error": "not_found"})
                return
            if not voice_api_authorized(self.headers, token):
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length <= 0 or length > 65536:
                self.send_json(400, {"ok": False, "error": "invalid_body_size"})
                return
            body = self.rfile.read(length)
            agent, message, error = parse_api_message_payload(
                body,
                self.headers.get("Content-Type", ""),
            )
            if error:
                self.send_json(400, {"ok": False, "error": error})
                return
            result = route_api_message_to_tmux(agent, message, commands)
            self.send_json(200 if result.get("ok") else 400, result)

    return VoiceApiHandler


def start_voice_api_server(config, commands):
    enabled = get_config_bool(
        config,
        "VOICE_API_ENABLED",
        "api_enabled",
        DEFAULT_CONFIG["api_enabled"],
    )
    if not enabled:
        log_voice_api_configuration(config, commands, enabled=False)
        return None
    host, port = get_voice_api_bind(config)
    try:
        server = http.server.ThreadingHTTPServer(
            (host, port),
            make_voice_api_handler(config, commands),
        )
    except OSError as exc:
        print(f"[api] unable to start local API on {host}:{port}: {exc}", file=sys.stderr)
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log_voice_api_configuration(config, commands, enabled=True)
    return server


def get_agent_completion_log_path(config):
    value = os.environ.get("VOICE_AGENT_COMPLETION_LOG")
    if value is None:
        value = config.get(
            "agent_completion_log",
            DEFAULT_CONFIG["agent_completion_log"],
        )
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in ("0", "false", "no", "none", "null", "off"):
        return None
    return os.path.expanduser(value)


def parse_agent_completion_record(line):
    try:
        record = json.loads(str(line or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    agent = str(record.get("agent") or "agent").strip() or "agent"
    status = str(record.get("status") or "done").strip() or "done"
    message = str(record.get("message") or "").strip()
    return {
        "agent": agent,
        "status": status,
        "message": message,
    }


def format_agent_completion_record(record):
    agent = record.get("agent") or "agent"
    status = record.get("status") or "done"
    message = record.get("message") or ""
    suffix = f": {message}" if message else ""
    return f"[agent-complete][{agent}] {status}{suffix}"


def start_agent_completion_log_tailer(config):
    path = get_agent_completion_log_path(config)
    if not path:
        return None

    retention_seconds = get_config_float(
        config,
        "VOICE_AGENT_COMPLETION_LOG_RETENTION_SECONDS",
        "agent_completion_log_retention_seconds",
        DEFAULT_CONFIG["agent_completion_log_retention_seconds"],
        minimum=0.0,
    )
    max_bytes = get_config_int(
        config,
        "VOICE_AGENT_COMPLETION_LOG_MAX_BYTES",
        "agent_completion_log_max_bytes",
        DEFAULT_CONFIG["agent_completion_log_max_bytes"],
        minimum=0,
    )
    poll_seconds = get_config_float(
        config,
        "VOICE_AGENT_COMPLETION_LOG_POLL_SECONDS",
        "agent_completion_log_poll_seconds",
        DEFAULT_CONFIG["agent_completion_log_poll_seconds"],
        minimum=0.05,
    )
    stop_event = threading.Event()

    def worker():
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                print(
                    f"[auto] unable to create agent completion log directory: {exc}",
                    file=sys.stderr,
                )
                return
        try:
            open(path, "a", encoding="utf-8").close()
        except OSError as exc:
            print(
                f"[auto] unable to create agent completion log {path}: {exc}",
                file=sys.stderr,
            )
            return

        position = os.path.getsize(path)
        print(f"[auto] agent completion log: {path}")
        last_trim = 0.0
        reported_read_error = False
        while not stop_event.is_set():
            try:
                size = os.path.getsize(path)
                if size < position:
                    position = 0
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        position = handle.tell()
                        record = parse_agent_completion_record(line)
                        if record:
                            print(format_agent_completion_record(record), flush=True)

                now = time.monotonic()
                if now - last_trim >= 60.0:
                    new_size = trim_auto_tmux_console_log(
                        path,
                        retention_seconds,
                        max_bytes,
                    )
                    if position > new_size:
                        position = new_size
                    last_trim = now
                reported_read_error = False
            except FileNotFoundError:
                position = 0
            except OSError as exc:
                if not reported_read_error:
                    print(
                        f"[auto] unable to read agent completion log {path}: {exc}",
                        file=sys.stderr,
                    )
                    reported_read_error = True
            stop_event.wait(poll_seconds)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event


def start_auto_tmux_console_log_tailer(config):
    path = get_auto_tmux_console_log_path(config)
    if not path:
        return None

    replay = get_config_bool(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_REPLAY",
        "auto_tmux_console_replay",
        DEFAULT_CONFIG["auto_tmux_console_replay"],
    )
    retention_seconds = get_config_float(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_RETENTION_SECONDS",
        "auto_tmux_console_retention_seconds",
        DEFAULT_CONFIG["auto_tmux_console_retention_seconds"],
        minimum=0.0,
    )
    max_bytes = get_config_int(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_MAX_BYTES",
        "auto_tmux_console_max_bytes",
        DEFAULT_CONFIG["auto_tmux_console_max_bytes"],
        minimum=0,
    )
    trim_interval = get_config_float(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_TRIM_INTERVAL_SECONDS",
        "auto_tmux_console_trim_interval_seconds",
        DEFAULT_CONFIG["auto_tmux_console_trim_interval_seconds"],
        minimum=1.0,
    )
    poll_seconds = get_config_float(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_POLL_SECONDS",
        "auto_tmux_console_poll_seconds",
        DEFAULT_CONFIG["auto_tmux_console_poll_seconds"],
        minimum=0.01,
    )
    idle_flush_seconds = get_config_float(
        config,
        "VOICE_AUTO_TMUX_CONSOLE_IDLE_FLUSH_SECONDS",
        "auto_tmux_console_idle_flush_seconds",
        DEFAULT_CONFIG["auto_tmux_console_idle_flush_seconds"],
        minimum=0.0,
    )
    summary_enabled = get_config_bool(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_ENABLED",
        "auto_tmux_summary_enabled",
        DEFAULT_CONFIG["auto_tmux_summary_enabled"],
    )
    summary_idle_seconds = get_config_float(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_IDLE_SECONDS",
        "auto_tmux_summary_idle_seconds",
        DEFAULT_CONFIG["auto_tmux_summary_idle_seconds"],
        minimum=0.0,
    )
    summary_lines = get_config_int(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_LINES",
        "auto_tmux_summary_lines",
        DEFAULT_CONFIG["auto_tmux_summary_lines"],
        minimum=1,
    )
    summary_max_chars = get_config_int(
        config,
        "VOICE_AUTO_TMUX_SUMMARY_MAX_CHARS",
        "auto_tmux_summary_max_chars",
        DEFAULT_CONFIG["auto_tmux_summary_max_chars"],
        minimum=200,
    )
    stop_event = threading.Event()

    def worker():
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                print(
                    f"[auto] unable to create tmux console log directory: {exc}",
                    file=sys.stderr,
                )
                return
        try:
            open(path, "a", encoding="utf-8").close()
        except OSError as exc:
            print(
                f"[auto] unable to create tmux console log {path}: {exc}",
                file=sys.stderr,
            )
            return

        if replay:
            trim_auto_tmux_console_log(
                path,
                retention_seconds,
                max_bytes,
            )
            position = 0
        else:
            try:
                position = os.path.getsize(path)
            except OSError:
                position = 0

        print(f"[auto] tmux console log: {path}")
        if summary_enabled:
            print(
                "[auto] tmux summaries enabled: "
                f"last {summary_lines} lines after {summary_idle_seconds:.1f}s idle"
            )
        last_trim = 0.0
        pending_output = []
        last_output_at = None
        agent_states = {}
        reported_read_error = False
        while not stop_event.is_set():
            try:
                saw_output = False
                updated_agents = set()
                size = os.path.getsize(path)
                if size < position:
                    position = 0
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        position = handle.tell()
                        _timestamp, payload = split_tmux_console_log_record(line)
                        if payload:
                            if summary_enabled:
                                agent_label, agent_payload = (
                                    extract_tmux_console_agent_payload(payload)
                                )
                                if agent_label:
                                    state = agent_states.setdefault(
                                        agent_label,
                                        {
                                            "chunks": [],
                                            "lines": [],
                                            "last_output_at": None,
                                            "last_digest": None,
                                        },
                                    )
                                    state["chunks"].append(agent_payload)
                                    updated_agents.add(agent_label)
                            elif line.startswith("{"):
                                pending_output.append(payload)
                            else:
                                pending_output.append(payload + "\n")
                            saw_output = True

                now = time.monotonic()
                if saw_output:
                    last_output_at = now
                for agent_label in updated_agents:
                    agent_states[agent_label]["last_output_at"] = now

                if summary_enabled:
                    for agent_label, state in list(agent_states.items()):
                        if not tmux_console_buffer_should_flush(
                            state["chunks"],
                            state["last_output_at"],
                            summary_idle_seconds,
                            now=now,
                        ):
                            continue
                        lines = tmux_console_payload_to_lines(
                            "".join(state["chunks"])
                        )
                        state["chunks"].clear()
                        if not lines:
                            continue
                        state["lines"].extend(lines)
                        trim_tmux_agent_lines(state["lines"], summary_lines)
                        summary_input_lines = list(state["lines"])[-summary_lines:]
                        while (
                            summary_max_chars > 0
                            and len("\n".join(summary_input_lines)) > summary_max_chars
                            and len(summary_input_lines) > 1
                        ):
                            summary_input_lines.pop(0)
                        command_record = get_tmux_sent_command(agent_label)
                        command_text = (
                            command_record.get("text") if command_record else ""
                        )
                        digest = tmux_summary_digest(
                            command_text,
                            summary_input_lines,
                        )
                        if digest == state.get("last_digest"):
                            continue
                        state["last_digest"] = digest
                        try:
                            summary = summarize_tmux_agent_output(
                                config,
                                agent_label,
                                command_text,
                                summary_input_lines,
                            )
                        except Exception as exc:
                            print(
                                f"\n[tmux-summary][{agent_label}] Command: "
                                f"{command_text or 'unknown'}"
                            )
                            print(
                                f"[tmux-summary][{agent_label}] Summary failed: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        print_tmux_summary(agent_label, command_text, summary)
                        dispatch_tmux_summary_webhook(
                            config,
                            agent_label,
                            command_text,
                            summary,
                        )
                elif tmux_console_buffer_should_flush(
                    pending_output,
                    last_output_at,
                    idle_flush_seconds,
                    now=now,
                ):
                    flush_tmux_console_buffer(pending_output)

                if now - last_trim >= trim_interval:
                    new_size = trim_auto_tmux_console_log(
                        path,
                        retention_seconds,
                        max_bytes,
                    )
                    if position > new_size:
                        position = new_size
                    last_trim = now
                reported_read_error = False
            except FileNotFoundError:
                position = 0
            except OSError as exc:
                if not reported_read_error:
                    print(
                        f"[auto] unable to read tmux console log {path}: {exc}",
                        file=sys.stderr,
                    )
                    reported_read_error = True
            stop_event.wait(poll_seconds)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event


def get_tmux_client_pid():
    if not shutil.which("tmux"):
        return None
    try:
        result = run_command(
            ["tmux", "display-message", "-p", "#{client_pid}"],
            timeout=0.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    pid = (result.stdout or "").strip().splitlines()
    if not pid:
        return None
    pid = pid[0].strip()
    return pid if pid.isdigit() else None


def find_x11_window_by_pid(pid):
    if not pid or not shutil.which("xdotool"):
        return None
    try:
        result = run_command(["xdotool", "search", "--pid", str(pid)], timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    windows = [line.strip() for line in (result.stdout or "").splitlines()]
    windows = [window for window in windows if window]
    if not windows:
        return None
    return windows[-1]


def focus_x11_window(window_id):
    window_id = str(window_id or "").strip()
    if not window_id:
        append_auto_focus_log(
            "focus-x11-command",
            error="empty-window-id",
            success=False,
        )
        return False
    if not shutil.which("xdotool"):
        append_auto_focus_log(
            "focus-x11-command",
            error="xdotool-not-found",
            target=window_id,
            success=False,
        )
        return False
    commands = (
        ["xdotool", "windowactivate", "--sync", window_id],
        ["xdotool", "windowactivate", window_id],
    )
    for argv in commands:
        try:
            result = run_command(argv, timeout=1.0)
        except subprocess.TimeoutExpired:
            append_auto_focus_log(
                "focus-x11-command",
                argv=argv,
                error="timeout",
                target=window_id,
                success=False,
            )
            continue
        except OSError as exc:
            append_auto_focus_log(
                "focus-x11-command",
                argv=argv,
                error=str(exc),
                target=window_id,
                success=False,
            )
            continue
        append_auto_focus_log(
            "focus-x11-command",
            argv=argv,
            returncode=result.returncode,
            stderr=summarize_log_text(result.stderr),
            stdout=summarize_log_text(result.stdout),
            success=result.returncode == 0,
            target=window_id,
        )
        if result.returncode == 0:
            delay = get_env_float(
                "VOICE_AUTO_REFOCUS_DELAY",
                AUTO_REFOCUS_DELAY_DEFAULT,
                minimum=0.0,
            )
            if delay:
                time.sleep(delay)
            return True
    return False


def focus_window_by_title(title):
    title = str(title or "").strip()
    if not title:
        return False
    if shutil.which("wmctrl"):
        try:
            result = run_command(["wmctrl", "-a", title], timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            return True
    if not shutil.which("xdotool"):
        return False
    try:
        result = run_command(["xdotool", "search", "--name", title], timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    windows = [line.strip() for line in (result.stdout or "").splitlines()]
    windows = [window for window in windows if window]
    if not windows:
        return False
    return focus_x11_window(windows[-1])


def is_gnome_desktop():
    values = (
        os.environ.get("XDG_CURRENT_DESKTOP", ""),
        os.environ.get("DESKTOP_SESSION", ""),
    )
    return any("gnome" in value.lower() or "ubuntu" in value.lower() for value in values)


def focus_gnome_app(app_id):
    app_id = str(app_id or "").strip()
    if not app_id or not shutil.which("gdbus"):
        return False
    try:
        result = run_command(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.gnome.Shell",
                "--object-path",
                "/org/gnome/Shell",
                "--method",
                "org.gnome.Shell.FocusApp",
                app_id,
            ],
            timeout=1.0,
        )
    except subprocess.TimeoutExpired:
        append_auto_focus_log(
            "focus-gnome-app",
            app_id=app_id,
            error="timeout",
            success=False,
        )
        return False
    except OSError as exc:
        append_auto_focus_log(
            "focus-gnome-app",
            app_id=app_id,
            error=str(exc),
            success=False,
        )
        return False
    append_auto_focus_log(
        "focus-gnome-app",
        app_id=app_id,
        returncode=result.returncode,
        stderr=summarize_log_text(result.stderr),
        stdout=summarize_log_text(result.stdout),
        success=result.returncode == 0,
    )
    return result.returncode == 0


def get_gnome_favorite_apps():
    if not shutil.which("gsettings"):
        return []
    try:
        result = run_command(
            ["gsettings", "get", "org.gnome.shell", "favorite-apps"],
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        append_auto_focus_log(
            "focus-gnome-favorites",
            returncode=result.returncode,
            stderr=summarize_log_text(result.stderr),
            stdout=summarize_log_text(result.stdout),
            success=False,
        )
        return []
    try:
        apps = ast.literal_eval((result.stdout or "").strip())
    except (SyntaxError, ValueError):
        append_auto_focus_log(
            "focus-gnome-favorites",
            error="parse-failed",
            stdout=summarize_log_text(result.stdout),
            success=False,
        )
        return []
    if not isinstance(apps, list):
        return []
    return [str(app) for app in apps]


def get_gnome_favorite_hotkey(app_id):
    configured = os.environ.get("VOICE_AUTO_FOCUS_HOTKEY", "").strip()
    if configured:
        if configured.lower() in ("0", "false", "no", "off"):
            return None
        return configured

    index_value = os.environ.get("VOICE_AUTO_GNOME_FAVORITE_INDEX", "").strip()
    if index_value.isdigit():
        index = int(index_value)
        if 1 <= index <= 9:
            return f"super+{index}"

    favorites = get_gnome_favorite_apps()
    if not favorites:
        return None
    try:
        index = favorites.index(app_id) + 1
    except ValueError:
        return None
    if 1 <= index <= 9:
        return f"super+{index}"
    return None


def get_gnome_terminal_focus_mode():
    value = os.environ.get("VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE", "").strip().lower()
    if value in ("off", "none", "disabled", "disable"):
        return "off"
    if value in ("app", "hotkey", "launch", "launch-first", "hotkey-first"):
        return value
    return "off"


def ydotoold_is_running():
    if not shutil.which("pgrep"):
        return False
    try:
        result = run_command(["pgrep", "-x", "ydotoold"], timeout=0.5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def ensure_ydotoold_running():
    if ydotoold_is_running():
        append_auto_focus_log("focus-ydotoold", action="already-running", success=True)
        return True
    if not shutil.which("ydotoold"):
        append_auto_focus_log(
            "focus-ydotoold",
            action="start",
            error="ydotoold-not-found",
            success=False,
        )
        return False
    socket_path = "/tmp/.ydotool_socket"
    try:
        if os.path.exists(socket_path):
            os.unlink(socket_path)
    except OSError as exc:
        append_auto_focus_log(
            "focus-ydotoold",
            action="remove-stale-socket",
            error=str(exc),
            success=False,
        )
    log_path = os.environ.get("VOICE_YDOTOOLD_LOG", "/tmp/ydotoold-workbench.log")
    try:
        log_handle = open(log_path, "ab")
        subprocess.Popen(
            ["ydotoold"],
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
        )
        log_handle.close()
    except OSError as exc:
        append_auto_focus_log(
            "focus-ydotoold",
            action="start",
            error=str(exc),
            success=False,
        )
        return False
    time.sleep(0.2)
    success = ydotoold_is_running()
    append_auto_focus_log(
        "focus-ydotoold",
        action="start",
        log_path=log_path,
        success=success,
    )
    return success


def focus_with_hotkey(hotkey):
    hotkey = str(hotkey or "").strip()
    if not hotkey:
        return False
    if not shutil.which("ydotool"):
        append_auto_focus_log(
            "focus-hotkey",
            error="ydotool-not-found",
            hotkey=hotkey,
            success=False,
        )
        return False
    ensure_ydotoold_running()
    try:
        result = run_command(["ydotool", "key", hotkey], timeout=1.0)
    except subprocess.TimeoutExpired:
        append_auto_focus_log(
            "focus-hotkey",
            error="timeout",
            hotkey=hotkey,
            success=False,
        )
        return False
    except OSError as exc:
        append_auto_focus_log(
            "focus-hotkey",
            error=str(exc),
            hotkey=hotkey,
            success=False,
        )
        return False
    detail = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    backend_available = "backend unavailable" not in detail
    success = result.returncode == 0 and backend_available
    append_auto_focus_log(
        "focus-hotkey",
        hotkey=hotkey,
        returncode=result.returncode,
        stderr=summarize_log_text(result.stderr),
        stdout=summarize_log_text(result.stdout),
        success=success,
    )
    if success:
        delay = get_env_float(
            "VOICE_AUTO_REFOCUS_DELAY",
            AUTO_REFOCUS_DELAY_DEFAULT,
            minimum=0.0,
        )
        if delay:
            time.sleep(delay)
        return True
    return False


def focus_gnome_terminal_launch():
    if not is_gnome_desktop():
        return False
    if not get_env_flag("VOICE_AUTO_GNOME_TERMINAL_LAUNCH_FALLBACK", default=True):
        append_auto_focus_log(
            "focus-gnome-terminal-launch",
            error="disabled",
            success=False,
        )
        return False
    terminal_command = os.environ.get(
        "VOICE_AUTO_GNOME_TERMINAL_COMMAND",
        "gnome-terminal",
    ).strip()
    if not terminal_command or not shutil.which(terminal_command):
        append_auto_focus_log(
            "focus-gnome-terminal-launch",
            error="terminal-command-not-found",
            terminal_command=terminal_command,
            success=False,
        )
        return False
    session = os.environ.get("VOICE_AUTO_TMUX_SESSION", "").strip()
    if session:
        try:
            tmux_result = run_command(["tmux", "has-session", "-t", session], timeout=1.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            append_auto_focus_log(
                "focus-gnome-terminal-launch",
                error=str(exc),
                session=session,
                success=False,
            )
            return False
        if tmux_result.returncode != 0:
            append_auto_focus_log(
                "focus-gnome-terminal-launch",
                returncode=tmux_result.returncode,
                session=session,
                stderr=summarize_log_text(tmux_result.stderr),
                stdout=summarize_log_text(tmux_result.stdout),
                success=False,
            )
            return False
        terminal_title = os.environ.get(
            "VOICE_AUTO_TERMINAL_TITLE",
            session,
        )
        argv = [
            terminal_command,
            "--title",
            terminal_title,
            "--",
            "tmux",
            "attach-session",
            "-t",
            session,
        ]
    else:
        argv = [terminal_command]
    try:
        result = run_command(argv, timeout=2.0)
    except subprocess.TimeoutExpired:
        append_auto_focus_log(
            "focus-gnome-terminal-launch",
            argv=argv,
            error="timeout",
            success=False,
        )
        return False
    except OSError as exc:
        append_auto_focus_log(
            "focus-gnome-terminal-launch",
            argv=argv,
            error=str(exc),
            success=False,
        )
        return False
    append_auto_focus_log(
        "focus-gnome-terminal-launch",
        argv=argv,
        returncode=result.returncode,
        stderr=summarize_log_text(result.stderr),
        stdout=summarize_log_text(result.stdout),
        success=result.returncode == 0,
    )
    if result.returncode == 0:
        delay = get_env_float(
            "VOICE_AUTO_REFOCUS_DELAY",
            AUTO_REFOCUS_DELAY_DEFAULT,
            minimum=0.0,
        )
        if delay:
            time.sleep(delay)
        return True
    return False


def focus_gnome_terminal_favorite():
    if not is_gnome_desktop():
        return False
    focus_mode = get_gnome_terminal_focus_mode()
    append_auto_focus_log("focus-method", method="gnome-focus-mode", target=focus_mode)
    if focus_mode == "off":
        return False
    app_id = os.environ.get(
        "VOICE_AUTO_GNOME_APP_ID",
        "org.gnome.Terminal.desktop",
    ).strip()
    if focus_mode == "app":
        if focus_gnome_app(app_id):
            append_auto_focus_log("focus-result", method="gnome-app", success=True)
            return True
        return False
    if focus_mode in ("launch", "launch-first"):
        if focus_gnome_terminal_launch():
            append_auto_focus_log(
                "focus-result",
                method="gnome-terminal-launch",
                success=True,
            )
            return True
        if focus_mode == "launch":
            return False
    if focus_mode not in ("hotkey", "hotkey-first", "launch-first"):
        return False
    hotkey = get_gnome_favorite_hotkey(app_id)
    append_auto_focus_log(
        "focus-method",
        method="gnome-favorite-hotkey",
        target=hotkey or "",
        success=bool(hotkey),
    )
    if hotkey and focus_with_hotkey(hotkey):
        append_auto_focus_log(
            "focus-result",
            method="gnome-favorite-hotkey",
            success=True,
        )
        return True
    if focus_mode == "hotkey-first" and focus_gnome_terminal_launch():
        append_auto_focus_log(
            "focus-result",
            method="gnome-terminal-launch",
            success=True,
        )
        return True
    return False


def focus_auto_terminal_window():
    if not get_env_flag("VOICE_AUTO_REFOCUS_TERMINAL", default=True):
        append_auto_focus_log("focus-disabled", reason="VOICE_AUTO_REFOCUS_TERMINAL")
        return False

    window_id = (
        os.environ.get("VOICE_AUTO_TERMINAL_WINDOW_ID")
        or os.environ.get("WINDOWID")
        or ""
    ).strip()
    title = os.environ.get("VOICE_AUTO_TERMINAL_WINDOW_TITLE", "").strip()
    append_auto_focus_log(
        "focus-start",
        display=os.environ.get("DISPLAY", ""),
        has_gdbus=bool(shutil.which("gdbus")),
        has_gsettings=bool(shutil.which("gsettings")),
        has_tmux=bool(shutil.which("tmux")),
        has_wmctrl=bool(shutil.which("wmctrl")),
        has_xdotool=bool(shutil.which("xdotool")),
        has_ydotool=bool(shutil.which("ydotool")),
        session_type=os.environ.get("XDG_SESSION_TYPE", ""),
        terminal_window_id=window_id,
        terminal_window_title=title,
        wayland_display=os.environ.get("WAYLAND_DISPLAY", ""),
    )
    if window_id:
        focused = focus_x11_window(window_id)
        append_auto_focus_log(
            "focus-method",
            method="window-id",
            target=window_id,
            success=focused,
        )
        if focused:
            append_auto_focus_log("focus-result", method="window-id", success=True)
            return True

    if title:
        focused = focus_window_by_title(title)
        append_auto_focus_log(
            "focus-method",
            method="window-title",
            target=title,
            success=focused,
        )
        if focused:
            append_auto_focus_log("focus-result", method="window-title", success=True)
            return True

    client_pid = get_tmux_client_pid()
    append_auto_focus_log(
        "focus-method",
        method="tmux-client-pid",
        target=client_pid or "",
        success=bool(client_pid),
    )
    if client_pid:
        window_id = find_x11_window_by_pid(client_pid)
        append_auto_focus_log(
            "focus-method",
            method="tmux-client-window",
            target=window_id or "",
            success=bool(window_id),
        )
        if window_id:
            focused = focus_x11_window(window_id)
            append_auto_focus_log(
                "focus-method",
                method="tmux-client-window-focus",
                target=window_id,
                success=focused,
            )
            if focused:
                append_auto_focus_log(
                    "focus-result",
                    method="tmux-client-window",
                    success=True,
                )
                return True

    if focus_gnome_terminal_favorite():
        return True

    append_auto_focus_log("focus-result", method="", success=False)
    return False


def run_command_checked(argv, input_text=None, label=None, timeout=None):
    try:
        result = run_command(argv, input_text=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        name = label or argv[0]
        print(f"[paste] {name} timed out.", file=sys.stderr)
        return False
    if result.returncode != 0:
        name = label or argv[0]
        detail = (result.stderr or result.stdout).strip()
        if name == "wtype" and "virtual keyboard protocol" in detail.lower():
            global WTYPE_AVAILABLE
            WTYPE_AVAILABLE = False
        if name == "ydotool" and "ydotoold" in detail.lower():
            global YDOTOOL_AVAILABLE
            YDOTOOL_AVAILABLE = False
        if detail:
            print(
                f"[paste] {name} failed ({result.returncode}): {detail}",
                file=sys.stderr,
            )
        else:
            print(
                f"[paste] {name} failed with code {result.returncode}.",
                file=sys.stderr,
            )
        return False
    return True


def run_auto_shell_command(command):
    label = command.get("label") or "command"
    argv = command.get("argv") or []
    if not argv:
        return False
    focus_success = focus_auto_terminal_window()
    append_auto_focus_log(
        "switch-start",
        argv=argv,
        focus_success=focus_success,
        label=label,
    )
    try:
        result = run_command(argv, timeout=2.0)
    except subprocess.TimeoutExpired:
        append_auto_focus_log("switch-result", label=label, success=False, error="timeout")
        print(f"[auto] switch command timed out: {label}", file=sys.stderr)
        return False
    except OSError as exc:
        append_auto_focus_log(
            "switch-result",
            label=label,
            success=False,
            error=str(exc),
        )
        print(f"[auto] switch command failed: {label}: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        append_auto_focus_log(
            "switch-result",
            label=label,
            returncode=result.returncode,
            stderr=summarize_log_text(result.stderr),
            stdout=summarize_log_text(result.stdout),
            success=False,
        )
        if detail:
            print(
                f"[auto] switch command failed for {label}: {detail}",
                file=sys.stderr,
            )
        else:
            print(
                f"[auto] switch command failed for {label} "
                f"with code {result.returncode}.",
                file=sys.stderr,
            )
        return False
    append_auto_focus_log(
        "switch-result",
        label=label,
        returncode=result.returncode,
        success=True,
    )
    message = command.get("success_message")
    if message:
        print(message)
    else:
        print(f"[auto] switched to {label}.")
    if command.get("exit_after"):
        raise SystemExit(0)
    return True


def make_tmux_send_buffer_name():
    return (
        f"voice-workbench-{os.getpid()}-"
        f"{threading.get_ident()}-{time.monotonic_ns()}"
    )


def send_text_to_tmux_target(command, text):
    target = str(command.get("tmux_send_target") or "").strip()
    label = command.get("label") or target or "tmux target"
    text = str(text or "")
    if not target:
        append_auto_focus_log(
            "tmux-send-result",
            error="missing-target",
            label=label,
            success=False,
        )
        return False
    if not shutil.which("tmux"):
        append_auto_focus_log(
            "tmux-send-result",
            error="tmux-not-found",
            label=label,
            target=target,
            success=False,
        )
        print(f"[auto] tmux not found; cannot send queued text to {label}.", file=sys.stderr)
        return False

    buffer_name = make_tmux_send_buffer_name()
    append_auto_focus_log(
        "tmux-send-start",
        label=label,
        target=target,
        text_length=len(text),
    )
    commands = []
    if text:
        commands.extend(
            [
                ["tmux", "set-buffer", "-b", buffer_name, "--", text],
                ["tmux", "paste-buffer", "-d", "-b", buffer_name, "-t", target],
            ]
        )
    enter_delay = get_submit_enter_delay()
    commands.append(["tmux", "send-keys", "-t", target, "C-m"])

    for index, argv in enumerate(commands):
        if enter_delay and index == len(commands) - 1:
            time.sleep(enter_delay)
        try:
            result = run_command(argv, timeout=2.0)
        except subprocess.TimeoutExpired:
            append_auto_focus_log(
                "tmux-send-result",
                argv=argv,
                error="timeout",
                label=label,
                target=target,
                success=False,
            )
            print(f"[auto] tmux send timed out for {label}.", file=sys.stderr)
            return False
        except OSError as exc:
            append_auto_focus_log(
                "tmux-send-result",
                argv=argv,
                error=str(exc),
                label=label,
                target=target,
                success=False,
            )
            print(f"[auto] tmux send failed for {label}: {exc}", file=sys.stderr)
            return False
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            append_auto_focus_log(
                "tmux-send-result",
                argv=argv,
                label=label,
                returncode=result.returncode,
                stderr=summarize_log_text(result.stderr),
                stdout=summarize_log_text(result.stdout),
                target=target,
                success=False,
            )
            if detail:
                print(f"[auto] tmux send failed for {label}: {detail}", file=sys.stderr)
            else:
                print(
                    f"[auto] tmux send failed for {label} with code {result.returncode}.",
                    file=sys.stderr,
                )
            return False

    append_auto_focus_log(
        "tmux-send-result",
        label=label,
        target=target,
        text_length=len(text),
        success=True,
    )
    if text:
        record_tmux_sent_command(label, target, text)
    return True


def get_paste_mode():
    value = os.environ.get("VOICE_PASTE_MODE", "").strip().lower()
    if value:
        if value in ("auto", "hotkey", "type", "clipboard"):
            return value
        return PASTE_MODE_DEFAULT
    return PASTE_MODE_DEFAULT


def get_paste_delay():
    value = os.environ.get("VOICE_PASTE_DELAY", "").strip()
    if not value:
        return float(PASTE_DELAY_DEFAULT)
    try:
        delay = float(value)
    except ValueError:
        return float(PASTE_DELAY_DEFAULT)
    return max(0.0, delay)


def get_submit_enter_delay():
    value = os.environ.get("VOICE_SUBMIT_ENTER_DELAY", "").strip()
    if not value:
        value = os.environ.get("VOICE_SUBMIT_ENTER_DELAY_SECONDS", "").strip()
    if not value:
        value = str(DEFAULT_CONFIG.get("submit_enter_delay_seconds", 0.5))
    try:
        delay = float(value)
    except ValueError:
        delay = float(DEFAULT_CONFIG.get("submit_enter_delay_seconds", 0.5))
    return max(0.0, delay)


def set_clipboard(text):
    if shutil.which("wl-copy"):
        log_paste_debug("clipboard: using wl-copy")
        return run_command_checked(
            ["wl-copy"], input_text=text, label="wl-copy", timeout=1.0
        )
    if shutil.which("xclip"):
        log_paste_debug("clipboard: using xclip")
        return run_command_checked(
            ["xclip", "-selection", "clipboard"],
            input_text=text,
            label="xclip",
        )
    if shutil.which("xsel"):
        log_paste_debug("clipboard: using xsel")
        return run_command_checked(
            ["xsel", "--clipboard", "--input"],
            input_text=text,
            label="xsel",
        )
    return False


def paste_via_hotkey():
    if YDOTOOL_AVAILABLE and shutil.which("ydotool"):
        log_paste_debug("hotkey: using ydotool")
        return run_command_checked(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            label="ydotool",
        )
    if WTYPE_AVAILABLE and shutil.which("wtype"):
        log_paste_debug("hotkey: using wtype")
        return run_command_checked(
            ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
            label="wtype",
        )
    if shutil.which("xdotool"):
        log_paste_debug("hotkey: using xdotool")
        return run_command_checked(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            label="xdotool",
        )
    log_paste_debug("hotkey: no tool available")
    return False


def type_text(text):
    if YDOTOOL_AVAILABLE and shutil.which("ydotool"):
        log_paste_debug("type: using ydotool")
        return run_command_checked(
            ["ydotool", "type", "--file", "-"],
            input_text=text,
            label="ydotool",
        )
    if WTYPE_AVAILABLE and shutil.which("wtype"):
        log_paste_debug("type: using wtype")
        return run_command_checked(["wtype", text], label="wtype")
    log_paste_debug("type: no tool available")
    return False


def type_text_and_submit(text):
    enter_delay = get_submit_enter_delay()
    if text and not type_text(text):
        return False
    if enter_delay:
        log_paste_debug(f"submit: enter delay={enter_delay}")
        time.sleep(enter_delay)
    log_paste_debug("submit: typing newline")
    return type_text("\n")


def press_enter():
    if WTYPE_AVAILABLE and shutil.which("wtype"):
        log_paste_debug("enter: using wtype")
        if run_command_checked(["wtype", "-k", "Return"], label="wtype"):
            return True
    if YDOTOOL_AVAILABLE and shutil.which("ydotool"):
        log_paste_debug("enter: using ydotool")
        return run_command_checked(
            ["ydotool", "key", "--delay", "100", "--key-delay", "50", "KEY_ENTER"],
            label="ydotool",
        )
    if shutil.which("xdotool"):
        log_paste_debug("enter: using xdotool")
        return run_command_checked(
            ["xdotool", "key", "--clearmodifiers", "Return"],
            label="xdotool",
        )
    log_paste_debug("enter: no tool available")
    return False


def paste_text(text):
    mode = get_paste_mode()
    delay = get_paste_delay()
    PASTE_IN_PROGRESS.set()
    log_paste_debug(f"mode={mode} delay={delay}")
    log_paste_debug(
        "tools "
        f"wtype={bool(WTYPE_AVAILABLE and shutil.which('wtype'))} "
        f"ydotool={bool(YDOTOOL_AVAILABLE and shutil.which('ydotool'))} "
        f"xdotool={bool(shutil.which('xdotool'))} "
        f"wl-copy={bool(shutil.which('wl-copy'))} "
        f"xclip={bool(shutil.which('xclip'))} "
        f"xsel={bool(shutil.which('xsel'))}"
    )
    try:
        clipboard_ok = None
        if mode != "type":
            clipboard_ok = set_clipboard(text)
            log_paste_debug(
                f"clipboard={'ok' if clipboard_ok else 'fail'}"
            )

        if mode == "clipboard":
            return bool(clipboard_ok)

        if delay:
            time.sleep(delay)

        if mode == "type":
            result = type_text(text)
            log_paste_debug(f"type result={'ok' if result else 'fail'}")
            return result
        if mode == "hotkey":
            result = paste_via_hotkey()
            log_paste_debug(f"hotkey result={'ok' if result else 'fail'}")
            return result
        if paste_via_hotkey():
            log_paste_debug("auto: hotkey ok")
            return True
        log_paste_debug("auto: hotkey failed, trying type")
        result = type_text(text)
        log_paste_debug(f"auto: type result={'ok' if result else 'fail'}")
        return result
    finally:
        PASTE_IN_PROGRESS.clear()


def paste_transcript_text(text, history_path=None, correction=None):
    text, send_enter = split_trailing_submit_command(text)
    if not text and not send_enter:
        return False, text, send_enter

    if send_enter and get_paste_mode() == "type":
        delay = get_paste_delay()
        PASTE_IN_PROGRESS.set()
        try:
            if delay:
                time.sleep(delay)
            if type_text_and_submit(text):
                append_transcript_history(text, history_path, correction=correction)
                return True, text, send_enter
            return False, text, send_enter
        finally:
            PASTE_IN_PROGRESS.clear()

    pasted = True
    if text:
        pasted = paste_text(text)
        if not pasted:
            return False, text, send_enter

    if send_enter:
        delay = get_paste_delay()
        if delay:
            time.sleep(delay)
        PASTE_IN_PROGRESS.set()
        try:
            if not press_enter():
                return False, text, send_enter
        finally:
            PASTE_IN_PROGRESS.clear()

    append_transcript_history(text, history_path, correction=correction)
    return True, text, send_enter


def get_transcript_history_path(config):
    value = os.environ.get("VOICE_TRANSCRIPT_HISTORY_PATH")
    if value is None:
        value = config.get("transcript_history_path")
    if value is None:
        return None
    path = str(value).strip()
    if path.lower() in ("", "0", "false", "no", "none", "null", "off"):
        return None
    return path


def get_run_mode(config):
    value = os.environ.get("VOICE_RUN_MODE") or config.get("run_mode", "hotkey")
    mode = str(value or "hotkey").strip().lower()
    if mode in ("auto", "automatic", "handsfree", "hands-free"):
        return "auto"
    return "hotkey"


def get_auto_pause_hotkey(config):
    value = os.environ.get("VOICE_AUTO_PAUSE_HOTKEY")
    if value is None:
        value = config.get("auto_pause_hotkey", "ctrl")
    value = str(value or "").strip().lower()
    if value in ("", "0", "false", "no", "none", "null", "off"):
        return None
    return value


def get_config_float(config, env_name, config_name, default, minimum=None):
    value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    return result


def samples_have_voice(samples, threshold):
    audio = mix_audio_to_mono(samples)
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return rms >= threshold


def get_config_int(config, env_name, config_name, default, minimum=None):
    value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_name, default)
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    return result


def get_config_bool(config, env_name, config_name, default=False):
    value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_name, default)
    parsed = parse_config_bool(value)
    if parsed is None:
        return bool(default)
    return bool(parsed)


def get_config_string(config, env_name, config_name, default=None):
    value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_name, default)
    if value is None:
        return None
    return str(value).strip()


def build_sherpa_vad(config, sample_rate):
    backend = (
        get_config_string(config, "VOICE_AUTO_VAD_BACKEND", "auto_vad_backend", "rms")
        or "rms"
    ).lower()
    if backend not in ("sherpa", "silero", "silero-vad", "sherpa-vad"):
        return None
    if int(sample_rate) != 16000:
        print(
            "[auto] Sherpa VAD requires 16000 Hz audio; using RMS VAD.",
            file=sys.stderr,
        )
        return None
    model_path = get_config_string(
        config,
        "VOICE_AUTO_SHERPA_VAD_MODEL",
        "auto_sherpa_vad_model",
        "models/silero_vad.onnx",
    )
    if not model_path:
        return None
    model_path = os.path.expanduser(model_path)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(
            f"[auto] Sherpa VAD model not found at '{model_path}'; using RMS VAD.",
            file=sys.stderr,
        )
        return None
    try:
        import sherpa_onnx

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.sample_rate = int(sample_rate)
        vad_config.provider = (
            get_config_string(
                config,
                "VOICE_AUTO_SHERPA_VAD_PROVIDER",
                "auto_sherpa_vad_provider",
                "cpu",
            )
            or "cpu"
        )
        vad_config.num_threads = get_config_int(
            config,
            "VOICE_AUTO_SHERPA_VAD_NUM_THREADS",
            "auto_sherpa_vad_num_threads",
            1,
            minimum=1,
        )
        vad_config.silero_vad.model = model_path
        vad_config.silero_vad.threshold = get_config_float(
            config,
            "VOICE_AUTO_SHERPA_VAD_THRESHOLD",
            "auto_sherpa_vad_threshold",
            0.5,
            minimum=0.01,
        )
        vad_config.silero_vad.min_speech_duration = get_config_float(
            config,
            "VOICE_AUTO_SHERPA_VAD_MIN_SPEECH_SECONDS",
            "auto_sherpa_vad_min_speech_seconds",
            0.05,
            minimum=0.01,
        )
        vad_config.silero_vad.min_silence_duration = get_config_float(
            config,
            "VOICE_AUTO_SHERPA_VAD_MIN_SILENCE_SECONDS",
            "auto_sherpa_vad_min_silence_seconds",
            0.2,
            minimum=0.01,
        )
        detector = sherpa_onnx.VoiceActivityDetector(vad_config, 30)
    except Exception as exc:
        print(f"[auto] Sherpa VAD unavailable: {exc}; using RMS VAD.", file=sys.stderr)
        return None
    print(f"[auto] using Sherpa VAD: {model_path}")
    return detector


def get_auto_initial_scan_sample(total_samples, pre_roll_samples):
    return max(0, int(total_samples) - max(0, int(pre_roll_samples)))


def click_mouse_left():
    if YDOTOOL_AVAILABLE and shutil.which("ydotool"):
        log_paste_debug("click: using ydotool")
        return run_command_checked(
            ["ydotool", "click", "1"],
            label="ydotool",
        )
    if shutil.which("xdotool"):
        log_paste_debug("click: using xdotool")
        return run_command_checked(
            ["xdotool", "click", "1"],
            label="xdotool",
        )
    log_paste_debug("click: no tool available")
    return False


def main():
    config_path = os.environ.get("VOICE_HOTKEY_CONFIG", "config.json")
    config = load_config(config_path)
    global PASTE_DEBUG_DEFAULT
    global PASTE_DELAY_DEFAULT
    global PASTE_MODE_DEFAULT
    if "paste_debug" in config:
        PASTE_DEBUG_DEFAULT = bool(config.get("paste_debug"))
    if "paste_delay" in config:
        try:
            PASTE_DELAY_DEFAULT = float(config.get("paste_delay") or 0.0)
        except (TypeError, ValueError):
            PASTE_DELAY_DEFAULT = 0.0
    if "paste_mode" in config:
        mode_value = str(config.get("paste_mode") or "").strip().lower()
        if mode_value in ("auto", "hotkey", "type", "clipboard"):
            PASTE_MODE_DEFAULT = mode_value
    sample_rate = int(config.get("sample_rate", 16000))
    channels = int(config.get("channels", 1))
    run_mode = get_run_mode(config)
    transcript_history_path = get_transcript_history_path(config)
    auto_pause_hotkey = (
        get_auto_pause_hotkey(config) if run_mode == "auto" else None
    )
    hotkey_display = "auto"

    if run_mode == "hotkey" or auto_pause_hotkey:
        try:
            backend = get_hotkey_backend()
        except ValueError as exc:
            if run_mode == "hotkey":
                print(f"[hotkey] {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"[auto] pause hotkey disabled: {exc}", file=sys.stderr)
            backend = None
    else:
        backend = None

    interactive = sys.stdin.isatty()
    if run_mode == "auto":
        if interactive and backend == "evdev":
            step_total = 2
        elif interactive:
            step_total = 1
        else:
            step_total = 0
    elif interactive and backend == "evdev":
        step_total = 3
    elif interactive:
        step_total = 2
    else:
        step_total = 0
    step_index = 1 if step_total else 0
    step_label = f"Step {step_index}/{step_total}:" if step_total else None

    audio_device = prompt_for_device(config_path, config, step_label=step_label)
    if step_total:
        step_index += 1

    evdev_device = None
    if backend == "evdev" and step_total and (
        run_mode == "hotkey" or auto_pause_hotkey
    ):
        step_label = f"Step {step_index}/{step_total}:"
        evdev_device = prompt_for_evdev_device(
            config_path, config, step_label=step_label
        )
        step_index += 1

    hotkey_name = None
    if run_mode == "hotkey":
        step_label = f"Step {step_index}/{step_total}:" if step_total else None
        hotkey_name = prompt_for_hotkey(
            config_path,
            config,
            backend,
            step_label=step_label,
            device=evdev_device,
        )
        hotkey_display = normalize_hotkey_name(hotkey_name)

    transcribe_audio, transcribe_backend, _transcribe_model_label = (
        resolve_initial_transcriber(config, sample_rate, channels)
    )
    fallback_backend = get_fallback_backend(config)
    fallback_transcriber = {"value": None}
    fallback_transcriber_lock = threading.Lock()
    backend_locks = {}
    backend_locks_lock = threading.Lock()

    def get_busy_fallback_transcriber():
        if (
            not fallback_backend
            or fallback_backend == transcribe_backend
        ):
            return None, None
        if fallback_transcriber["value"] is None:
            with fallback_transcriber_lock:
                if fallback_transcriber["value"] is None:
                    fallback_transcriber["value"] = build_transcriber(
                        config,
                        sample_rate,
                        channels,
                        backend_override=fallback_backend,
                        allow_fallback=False,
                    )
        fallback_audio, fallback_name, _fallback_model_label = fallback_transcriber[
            "value"
        ]
        return fallback_audio, fallback_name

    def get_backend_lock(backend_name):
        with backend_locks_lock:
            if backend_name not in backend_locks:
                backend_locks[backend_name] = threading.Lock()
            return backend_locks[backend_name]

    recorder = Recorder(
        sample_rate=sample_rate, channels=channels, device=audio_device
    )

    state = {
        "active": False,
        "busy_count": 0,
        "busy_notice": False,
        "record_backend": None,
        "session": None,
        "next_job_id": 0,
        "drop_primary_before_job_id": 0,
    }
    auto_control = {
        "paused": False,
        "event": threading.Event(),
        "lock": threading.Lock(),
    }
    meter_state = {"stop": threading.Event(), "thread": None}
    backend_unavailable_until = {}
    try:
        chunk_seconds = max(
            1.0, float(config.get("chunk_transcribe_seconds") or 20.0)
        )
    except (TypeError, ValueError):
        chunk_seconds = 20.0
    try:
        pause_window_seconds = max(
            0.0, float(config.get("chunk_pause_window_seconds") or 1.0)
        )
    except (TypeError, ValueError):
        pause_window_seconds = 1.0
    try:
        vad_frame_ms = max(
            10.0, float(config.get("chunk_vad_frame_ms") or 30.0)
        )
    except (TypeError, ValueError):
        vad_frame_ms = 30.0
    try:
        vad_silence_ms = max(
            60.0, float(config.get("chunk_vad_silence_ms") or 240.0)
        )
    except (TypeError, ValueError):
        vad_silence_ms = 240.0
    try:
        vad_threshold = max(
            0.0001, float(config.get("chunk_vad_threshold") or 0.012)
        )
    except (TypeError, ValueError):
        vad_threshold = 0.012
    transcribe_request_timeout = get_transcribe_request_timeout(config)
    try:
        backend_retry_cooldown = max(
            0.0, float(config.get("backend_retry_cooldown") or 300.0)
        )
    except (TypeError, ValueError):
        backend_retry_cooldown = 300.0
    chunked_transcription = bool(config.get("chunked_transcription", True))
    chunk_poll_seconds = 0.1
    chunk_flush_samples = int(
        (chunk_seconds + pause_window_seconds) * sample_rate
    )
    chunk_target_samples = int(chunk_seconds * sample_rate)
    auto_vad_threshold = get_config_float(
        config,
        "VOICE_AUTO_VAD_THRESHOLD",
        "auto_vad_threshold",
        vad_threshold,
        minimum=0.0001,
    )
    auto_start_speech_ms = get_config_float(
        config,
        "VOICE_AUTO_START_SPEECH_MS",
        "auto_start_speech_ms",
        60.0,
        minimum=10.0,
    )
    auto_trigger_silence_seconds = get_config_float(
        config,
        "VOICE_AUTO_TRIGGER_SILENCE_SECONDS",
        "auto_trigger_silence_seconds",
        2.0,
        minimum=0.2,
    )
    auto_trigger_probe_seconds = get_config_float(
        config,
        "VOICE_AUTO_TRIGGER_PROBE_SECONDS",
        "auto_trigger_probe_seconds",
        0.5,
        minimum=0.2,
    )
    auto_trigger_min_probe_seconds = get_config_float(
        config,
        "VOICE_AUTO_TRIGGER_MIN_PROBE_SECONDS",
        "auto_trigger_min_probe_seconds",
        1.0,
        minimum=0.2,
    )
    auto_trigger_probe_window_seconds = get_config_float(
        config,
        "VOICE_AUTO_TRIGGER_PROBE_WINDOW_SECONDS",
        "auto_trigger_probe_window_seconds",
        1.5,
        minimum=0.5,
    )
    auto_trigger_arm_timeout_seconds = get_config_float(
        config,
        "VOICE_AUTO_TRIGGER_ARM_TIMEOUT_SECONDS",
        "auto_trigger_arm_timeout_seconds",
        8.0,
        minimum=0.0,
    )
    auto_trigger_word = str(
        os.environ.get("VOICE_AUTO_TRIGGER_WORD")
        or config.get("auto_trigger_word", "agent")
        or "agent"
    ).strip()
    auto_trigger_alias_value = os.environ.get("VOICE_AUTO_TRIGGER_ALIASES")
    if auto_trigger_alias_value is None:
        auto_trigger_alias_value = config.get("auto_trigger_aliases", [])
    auto_trigger_aliases = parse_word_list(auto_trigger_alias_value)
    auto_shell_commands = build_auto_tmux_switch_commands(config)
    auto_command_labels = sorted(
        {
            str(command.get("label") or "").strip()
            for command in auto_shell_commands.values()
            if command.get("label")
        }
    )
    transcript_correction_command_labels = (
        auto_command_labels + [auto_trigger_word] + auto_trigger_aliases
    )
    auto_pre_roll_seconds = get_config_float(
        config,
        "VOICE_AUTO_PRE_ROLL_SECONDS",
        "auto_pre_roll_seconds",
        1.5,
        minimum=0.0,
    )
    auto_poll_seconds = get_config_float(
        config,
        "VOICE_AUTO_POLL_SECONDS",
        "auto_poll_seconds",
        0.05,
        minimum=0.01,
    )
    auto_max_utterance_seconds = get_config_float(
        config,
        "VOICE_AUTO_MAX_UTTERANCE_SECONDS",
        "auto_max_utterance_seconds",
        90.0,
        minimum=1.0,
    )
    auto_vad_detector = build_sherpa_vad(config, sample_rate)
    start_transcript_correction_server_background(config)
    signal_voice_ready(run_mode, transcribe_backend, _transcribe_model_label)
    tmux_console_log_tailer = None
    agent_completion_log_tailer = None
    voice_api_server = None
    if run_mode == "auto":
        log_tmux_summary_webhook_configuration(config)
        tmux_console_log_tailer = start_auto_tmux_console_log_tailer(config)
        agent_completion_log_tailer = start_agent_completion_log_tailer(config)
        voice_api_server = start_voice_api_server(config, auto_shell_commands)

    def render_meter(level, width=24):
        filled = int(level * width)
        return "|" * filled + " " * (width - filled)

    def meter_loop():
        while not meter_state["stop"].is_set():
            level = recorder.get_level()
            bar = render_meter(level)
            print(f"\r[audio] {bar}", end="", flush=True)
            meter_state["stop"].wait(0.1)
        print()

    def start_meter():
        if meter_state["thread"] and meter_state["thread"].is_alive():
            return
        meter_state["stop"].clear()
        meter_state["thread"] = threading.Thread(
            target=meter_loop, daemon=True
        )
        meter_state["thread"].start()

    def stop_meter():
        meter_state["stop"].set()
        thread = meter_state["thread"]
        if thread:
            thread.join(timeout=0.25)
        meter_state["thread"] = None

    def is_backend_temporarily_unavailable(backend_name):
        deadline = backend_unavailable_until.get(backend_name, 0.0)
        return deadline > time.monotonic()

    def mark_backend_unavailable(backend_name, reason):
        if backend_retry_cooldown <= 0:
            return
        deadline = time.monotonic() + backend_retry_cooldown
        backend_unavailable_until[backend_name] = max(
            deadline,
            backend_unavailable_until.get(backend_name, 0.0),
        )
        print(
            f"[hotkey] backend '{backend_name}' unavailable for "
            f"{backend_retry_cooldown:.0f}s: {reason}",
            file=sys.stderr,
        )

    def run_transcribe_request(transcribe_fn, transcribe_name, wav_path):
        if transcribe_request_timeout <= 0:
            return transcribe_fn(wav_path)
        result = {"text": "", "error": None}

        def worker():
            try:
                result["text"] = transcribe_fn(wav_path)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=transcribe_request_timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"backend '{transcribe_name}' timed out after "
                f"{transcribe_request_timeout:.0f}s"
            )
        if result["error"] is not None:
            raise result["error"]
        return result["text"]

    def report_paste_failure(text):
        print(
            "[hotkey] unable to paste automatically.",
            file=sys.stderr,
        )
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        is_wayland = session == "wayland" or os.environ.get(
            "WAYLAND_DISPLAY"
        )
        if is_wayland:
            missing = []
            if not WTYPE_AVAILABLE:
                print(
                    "[hotkey] wtype blocked by compositor. "
                    "Try VOICE_PASTE_MODE=clipboard or use a "
                    "virtual keyboard-capable compositor.",
                    file=sys.stderr,
                )
            if not YDOTOOL_AVAILABLE:
                print(
                    "[hotkey] ydotoold is not running. "
                    "Start it to enable ydotool-based typing.",
                    file=sys.stderr,
                )
            if not shutil.which("wtype"):
                missing.append("wtype")
            if not shutil.which("ydotool"):
                missing.append("ydotool")
            if not shutil.which("wl-copy"):
                missing.append("wl-clipboard")
            if missing:
                print(
                    "[hotkey] install for Wayland: " + " ".join(missing),
                    file=sys.stderr,
                )
        else:
            missing = []
            if not shutil.which("xdotool"):
                missing.append("xdotool")
            if not (shutil.which("xclip") or shutil.which("xsel")):
                missing.append("xclip/xsel")
            if missing:
                print(
                    "[hotkey] install for X11: " + " ".join(missing),
                    file=sys.stderr,
                )
        print(text)

    def transcribe_sample_block(session_state, sample_block, chunk_index):
        if sample_block.size == 0:
            return ""
        log_prefix = session_state.get("log_prefix", "hotkey")
        transcribe_name = session_state["transcribe_name"]
        duration = len(sample_block) / float(sample_rate or 1)
        print(
            f"[{log_prefix}] transcribing chunk {chunk_index + 1} "
            f"({duration:.2f}s) with '{transcribe_name}'..."
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = handle.name
        try:
            write_wav(wav_path, sample_block, sample_rate, channels)
            print(
                f"[{log_prefix}] "
                f"{format_wav_stats(f'chunk {chunk_index + 1} wav', wav_path)}"
            )
            try:
                with get_backend_lock(transcribe_name):
                    text = run_transcribe_request(
                        session_state["transcribe_fn"],
                        transcribe_name,
                        wav_path,
                    )
            except Exception as exc:
                print(
                    f"[{log_prefix}] backend '{transcribe_name}' failed on chunk "
                    f"{chunk_index + 1}: {exc}",
                    file=sys.stderr,
                )
                mark_backend_unavailable(transcribe_name, exc)
                if transcribe_name != transcribe_backend:
                    return ""
                fallback_audio, fallback_name = get_busy_fallback_transcriber()
                if fallback_audio is None or fallback_name == transcribe_name:
                    return ""
                print(
                    f"[{log_prefix}] retrying chunk {chunk_index + 1} with fallback "
                    f"backend '{fallback_name}'..."
                )
                session_state["record_backend"] = "fallback"
                session_state["transcribe_fn"] = fallback_audio
                session_state["transcribe_name"] = fallback_name
                try:
                    with get_backend_lock(fallback_name):
                        text = run_transcribe_request(
                            fallback_audio,
                            fallback_name,
                            wav_path,
                        )
                except Exception as fallback_exc:
                    print(
                        f"[{log_prefix}] fallback backend '{fallback_name}' failed on "
                        f"chunk {chunk_index + 1}: {fallback_exc}",
                        file=sys.stderr,
                    )
                    mark_backend_unavailable(fallback_name, fallback_exc)
                    return ""
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        skip_model_correction = bool(
            session_state.get("skip_transcript_correction")
        ) and not transcript_correction_applies_to_probes(config)
        correction = correct_transcript_details(
            text,
            config,
            command_labels=transcript_correction_command_labels,
            skip_model=skip_model_correction,
        )
        text = correction["corrected_transcript"]
        if is_likely_bad_transcript(text):
            if text:
                print(f"[{log_prefix}] rejected suspicious transcript: {text!r}")
            return ""
        if text:
            session_state["last_transcript_correction"] = correction
            session_state.setdefault("transcript_corrections", []).append(correction)
            print(f"[{log_prefix}] chunk {chunk_index + 1} -> {text}")
        return text

    def transcribe_chunk_loop(session_state):
        try:
            while True:
                pending, _total = recorder.get_samples_since(
                    session_state["next_sample"]
                )
                pending_samples = len(pending)
                if pending_samples == 0:
                    if session_state["stop_event"].is_set():
                        break
                    session_state["stop_event"].wait(chunk_poll_seconds)
                    continue

                chunk_end = None
                if chunked_transcription:
                    chunk_end = choose_chunk_end_sample(
                        pending,
                        sample_rate,
                        chunk_seconds,
                        pause_window_seconds,
                        vad_frame_ms,
                        vad_silence_ms,
                        vad_threshold,
                    )
                    if chunk_end is None and pending_samples >= chunk_flush_samples:
                        chunk_end = min(chunk_target_samples, pending_samples)
                if (
                    chunk_end is None
                    and session_state["stop_event"].is_set()
                ):
                    if chunked_transcription and pending_samples > chunk_target_samples:
                        chunk_end = min(chunk_target_samples, pending_samples)
                    else:
                        chunk_end = pending_samples

                if chunk_end is None or chunk_end <= 0:
                    session_state["stop_event"].wait(chunk_poll_seconds)
                    continue

                chunk_samples = pending[:chunk_end]
                if chunk_samples.size == 0:
                    if session_state["stop_event"].is_set():
                        break
                    session_state["stop_event"].wait(chunk_poll_seconds)
                    continue

                chunk_index = session_state["chunk_count"]
                session_state["chunk_count"] += 1
                session_state["next_sample"] += len(chunk_samples)
                text = transcribe_sample_block(
                    session_state,
                    chunk_samples,
                    chunk_index,
                )
                session_state["chunk_results"].append(text)
        except Exception as exc:
            session_state["error"] = exc
        finally:
            session_state["done_event"].set()

    def finalize_recording_session(session_state, job_id):
        transcribe_name = session_state["transcribe_name"]
        try:
            session_state["stop_event"].set()
            if session_state["thread"] is not None:
                session_state["thread"].join()
            if session_state["error"] is not None:
                raise session_state["error"]
            text = " ".join(
                part for part in session_state["chunk_results"] if part
            ).strip()
            if (
                transcribe_name == transcribe_backend
                and job_id < state["drop_primary_before_job_id"]
            ):
                print(
                    "[hotkey] dropped stale primary transcript after a newer "
                    "fallback recording."
                )
                return
            if not text:
                print("[hotkey] no speech detected.")
                return
            final_correction = correct_transcript_details(
                text,
                config,
                command_labels=transcript_correction_command_labels,
            )
            text = final_correction["corrected_transcript"]
            history_correction = combine_transcript_correction_details(
                session_state.get("transcript_corrections", []),
                final_correction,
            )
            if text and config.get("paste_append_space"):
                text = text.rstrip() + " "
            if not text:
                print("[hotkey] no speech detected.")
                return
            paste_started = time.perf_counter()
            pasted, pasted_text, sent_enter = paste_transcript_text(
                text,
                transcript_history_path,
                correction=history_correction,
            )
            paste_ms = (time.perf_counter() - paste_started) * 1000
            if pasted_text:
                print(f"[hotkey] text ready: {pasted_text}")
            elif sent_enter:
                print("[hotkey] enter ready.")
            if not pasted:
                report_paste_failure(pasted_text or text)
            else:
                action = "pasted"
                if sent_enter and pasted_text:
                    action = "pasted and pressed enter"
                elif sent_enter:
                    action = "pressed enter"
                print(f"[hotkey] {action} in {paste_ms:.1f} ms: {pasted_text}")
        except Exception as exc:
            print(f"[hotkey] transcription failed: {exc}", file=sys.stderr)
        finally:
            recorder.clear_if_idle(session_state["recording_id"])
            state["busy_count"] = max(0, state["busy_count"] - 1)
            state["busy_notice"] = False
            if state["busy_count"] == 0:
                print(f"[hotkey] ready (hold '{hotkey_display}')")
            else:
                print(
                    "[hotkey] "
                    f"'{transcribe_name}' finished; "
                    f"{state['busy_count']} transcription(s) still running."
                )

    def handle_press():
        if state["active"]:
            return
        if PASTE_IN_PROGRESS.is_set():
            return
        record_backend = "primary"
        transcribe_fn = transcribe_audio
        transcribe_name = transcribe_backend
        primary_unavailable = is_backend_temporarily_unavailable(
            transcribe_backend
        )
        if primary_unavailable:
            fallback_audio, fallback_name = get_busy_fallback_transcriber()
            if fallback_audio is not None:
                record_backend = "fallback"
                transcribe_fn = fallback_audio
                transcribe_name = fallback_name
                print(
                    "[hotkey] primary backend temporarily unavailable; "
                    f"using fallback backend '{fallback_name}'"
                )
        elif state["busy_count"] > 0:
            fallback_audio, fallback_name = get_busy_fallback_transcriber()
            if fallback_audio is None:
                if not state["busy_notice"]:
                    state["busy_notice"] = True
                    print("[hotkey] busy transcribing, wait...")
                return
            record_backend = "fallback"
            if not state["busy_notice"]:
                state["busy_notice"] = True
                print(
                    "[hotkey] primary busy; using fallback backend "
                    f"'{fallback_name}'"
                )
            transcribe_fn = fallback_audio
            transcribe_name = fallback_name
        state["active"] = True
        state["record_backend"] = record_backend
        print(f"[hotkey] pressed '{hotkey_display}'")
        print("[hotkey] recording...")
        try:
            recording_id = recorder.start()
        except Exception:
            state["active"] = False
            state["record_backend"] = None
            raise
        state["session"] = {
            "record_backend": record_backend,
            "transcribe_fn": transcribe_fn,
            "transcribe_name": transcribe_name,
            "next_sample": 0,
            "chunk_count": 0,
            "chunk_results": [],
            "transcript_corrections": [],
            "error": None,
            "recording_id": recording_id,
            "stop_event": threading.Event(),
            "done_event": threading.Event(),
            "thread": None,
        }
        state["session"]["thread"] = threading.Thread(
            target=transcribe_chunk_loop,
            args=(state["session"],),
            daemon=True,
        )
        state["session"]["thread"].start()
        start_meter()

    def handle_release():
        if not state["active"]:
            return
        state["active"] = False
        stop_meter()
        print(f"[hotkey] released '{hotkey_display}'")
        session_state = state["session"]
        state["session"] = None
        captured_samples = recorder.get_total_samples()
        recorder.stop(return_samples=False)
        if session_state is None or captured_samples == 0:
            state["record_backend"] = None
            if session_state is not None:
                session_state["stop_event"].set()
                if session_state["thread"] is not None:
                    session_state["thread"].join(timeout=1.0)
                recorder.clear_if_idle(session_state["recording_id"])
            print("[hotkey] no audio captured.")
            return
        record_backend = session_state["record_backend"] or "primary"
        state["record_backend"] = None
        state["busy_count"] += 1
        state["next_job_id"] += 1
        job_id = state["next_job_id"]
        state["busy_notice"] = False
        if record_backend == "fallback":
            state["drop_primary_before_job_id"] = max(
                state["drop_primary_before_job_id"], job_id
            )
            print(
                "[hotkey] finalizing with fallback backend "
                f"'{session_state['transcribe_name']}'..."
            )
        else:
            print("[hotkey] finalizing transcript...")

        thread = threading.Thread(
            target=finalize_recording_session,
            args=(session_state, job_id),
            daemon=True,
        )
        thread.start()

    def is_auto_paused():
        with auto_control["lock"]:
            return bool(auto_control["paused"])

    def set_auto_paused(paused):
        with auto_control["lock"]:
            if auto_control["paused"] == paused:
                return
            auto_control["paused"] = paused
            auto_control["event"].set()
        if paused:
            if get_config_bool(
                config,
                "VOICE_API_ENABLED",
                "api_enabled",
                DEFAULT_CONFIG["api_enabled"],
            ):
                host, port = get_voice_api_bind(config)
                print(
                    "[auto] paused audio listening. API routing remains active: "
                    f"POST {build_voice_api_post_url(host, port)}. "
                    "Press Ctrl again to resume."
                )
            else:
                print("[auto] paused audio listening. Press Ctrl again to resume.")
        else:
            print("[auto] resumed audio listening.")

    def toggle_auto_pause():
        with auto_control["lock"]:
            paused = not auto_control["paused"]
        set_auto_paused(paused)

    def start_auto_pause_listener():
        if not auto_pause_hotkey or backend not in ("evdev", "pynput"):
            print("[auto] Ctrl pause/resume disabled.")
            return
        try:
            if backend == "evdev":
                pause_codes = parse_auto_pause_hotkey_evdev(auto_pause_hotkey)
                target = listen_hotkeys_evdev
                args = (
                    pause_codes,
                    toggle_auto_pause,
                    lambda: None,
                )
                kwargs = {"selected_device": evdev_device}
            else:
                pause_keys = parse_auto_pause_hotkey_pynput(auto_pause_hotkey)
                target = listen_hotkeys_pynput
                args = (
                    pause_keys,
                    toggle_auto_pause,
                    lambda: None,
                )
                kwargs = {}
        except Exception as exc:
            print(f"[auto] pause hotkey disabled: {exc}", file=sys.stderr)
            return

        thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()
        print(f"[auto] press {auto_pause_hotkey} to pause/resume listening.")

    def select_auto_transcriber():
        record_backend = "primary"
        transcribe_fn = transcribe_audio
        transcribe_name = transcribe_backend
        if is_backend_temporarily_unavailable(transcribe_backend):
            fallback_audio, fallback_name = get_busy_fallback_transcriber()
            if fallback_audio is not None:
                record_backend = "fallback"
                transcribe_fn = fallback_audio
                transcribe_name = fallback_name
                print(
                    "[auto] primary backend temporarily unavailable; "
                    f"using fallback backend '{fallback_name}'"
                )
        return {
            "record_backend": record_backend,
            "transcribe_fn": transcribe_fn,
            "transcribe_name": transcribe_name,
            "log_prefix": "auto",
        }

    auto_trigger_session = make_auto_trigger_session()

    def print_auto_waiting():
        if auto_shell_commands:
            print("[auto] waiting for: " + format_colored_detection_words(), flush=True)

    def click_auto_trigger_target(source="trigger"):
        print(f"[auto] trigger '{auto_trigger_word}' detected; clicking target...")
        focus_success = click_mouse_left()
        if not focus_success:
            print(
                "[auto] unable to left-click automatically; "
                "continuing with current focus.",
                file=sys.stderr,
            )
        arm_auto_trigger_session(
            auto_trigger_session,
            source,
            focus_success=focus_success,
        )
        return focus_success

    def reset_auto_vad():
        if auto_vad_detector is not None:
            try:
                auto_vad_detector.reset()
            except Exception as exc:
                print(f"[auto] Sherpa VAD reset failed: {exc}", file=sys.stderr)

    def auto_frame_has_voice(frame):
        if auto_vad_detector is None:
            return samples_have_voice(frame, auto_vad_threshold)
        try:
            audio = mix_audio_to_mono(frame)
            if audio.size == 0:
                return False
            auto_vad_detector.accept_waveform(audio)
            return bool(auto_vad_detector.is_speech_detected())
        except Exception as exc:
            print(f"[auto] Sherpa VAD failed: {exc}; using RMS VAD.", file=sys.stderr)
            return samples_have_voice(frame, auto_vad_threshold)

    def probe_auto_trigger(samples, chunk_index):
        if samples.size == 0:
            return False
        session_state = select_auto_transcriber()
        session_state["skip_transcript_correction"] = True
        text = transcribe_sample_block(session_state, samples, chunk_index).strip()
        if not text:
            return False
        command_prefix = safe_match_auto_shell_command_prefix(
            text,
            auto_shell_commands,
        )
        if command_prefix is not None:
            shell_command, _queued_text = command_prefix
            reset_auto_trigger_session(auto_trigger_session)
            if auto_shell_command_allowed(
                shell_command,
                session_state.get("last_transcript_correction"),
            ):
                run_auto_shell_command(shell_command)
            print_auto_waiting()
            return False
        shell_command = safe_match_auto_shell_command(text, auto_shell_commands)
        if shell_command is not None:
            reset_auto_trigger_session(auto_trigger_session)
            if auto_shell_command_allowed(
                shell_command,
                session_state.get("last_transcript_correction"),
            ):
                run_auto_shell_command(shell_command)
            print_auto_waiting()
            return False
        queued_text = extract_text_after_trigger_word(
            text,
            auto_trigger_word,
            auto_trigger_aliases,
        )
        if queued_text is None:
            print(f"[auto] probe did not hear '{auto_trigger_word}'.")
            return False
        return click_auto_trigger_target(source="probe")

    def wait_for_auto_utterance():
        frame_size = max(1, int(sample_rate * max(vad_frame_ms, 1) / 1000.0))
        start_voice_samples = max(
            frame_size,
            int(sample_rate * auto_start_speech_ms / 1000.0),
        )
        silence_samples = max(1, int(sample_rate * auto_trigger_silence_seconds))
        probe_samples = max(1, int(sample_rate * auto_trigger_probe_seconds))
        min_probe_samples = max(
            1,
            int(sample_rate * auto_trigger_min_probe_seconds),
        )
        probe_window_samples = max(
            min_probe_samples,
            int(sample_rate * auto_trigger_probe_window_seconds),
        )
        pre_roll_samples = int(sample_rate * auto_pre_roll_seconds)
        max_utterance_samples = int(sample_rate * auto_max_utterance_seconds)
        scan_sample = get_auto_initial_scan_sample(
            recorder.get_total_samples(),
            pre_roll_samples,
        )
        consecutive_voice_samples = 0
        utterance_start_sample = None
        last_voice_sample = None
        next_probe_sample = None
        probe_count = 0
        trigger_clicked = False
        reset_auto_vad()

        while True:
            if is_auto_paused():
                return None
            pending, _total = recorder.get_samples_since(scan_sample)
            if pending.size == 0:
                if auto_control["event"].wait(auto_poll_seconds):
                    auto_control["event"].clear()
                continue

            offset = 0
            while offset < len(pending):
                if is_auto_paused():
                    return None
                frame = pending[offset : offset + frame_size]
                if frame.size == 0:
                    break
                frame_end = scan_sample + offset + len(frame)
                has_voice = auto_frame_has_voice(frame)

                if utterance_start_sample is None:
                    if has_voice:
                        consecutive_voice_samples += len(frame)
                    else:
                        consecutive_voice_samples = 0
                    if consecutive_voice_samples >= start_voice_samples:
                        utterance_start_sample = max(
                            0,
                            frame_end
                            - consecutive_voice_samples
                            - pre_roll_samples,
                        )
                        last_voice_sample = frame_end
                        print(
                            "[auto] speech detected; listening for "
                            f"'{auto_trigger_word}' until "
                            f"{auto_trigger_silence_seconds:.1f}s of silence..."
                        )
                        next_probe_sample = frame_end + min_probe_samples
                else:
                    if has_voice:
                        last_voice_sample = frame_end
                    if (
                        not trigger_clicked
                        and next_probe_sample is not None
                        and frame_end >= next_probe_sample
                    ):
                        probe_start_sample = max(
                            utterance_start_sample,
                            frame_end - probe_window_samples,
                        )
                        samples, _ = recorder.get_samples_since(probe_start_sample)
                        length = max(0, frame_end - probe_start_sample)
                        trigger_clicked = probe_auto_trigger(
                            samples[:length],
                            probe_count,
                        )
                        probe_count += 1
                        next_probe_sample = frame_end + probe_samples
                    if (
                        last_voice_sample is not None
                        and frame_end - last_voice_sample >= silence_samples
                    ):
                        samples, _ = recorder.get_samples_since(
                            utterance_start_sample
                        )
                        length = max(0, frame_end - utterance_start_sample)
                        return samples[:length], trigger_clicked
                    if (
                        max_utterance_samples > 0
                        and frame_end - utterance_start_sample
                        >= max_utterance_samples
                    ):
                        print(
                            "[auto] max utterance length reached; "
                            "finalizing transcript..."
                        )
                        samples, _ = recorder.get_samples_since(
                            utterance_start_sample
                        )
                        return samples[:max_utterance_samples], trigger_clicked

                offset += len(frame)

            scan_sample += len(pending)
            if utterance_start_sample is None:
                recorder.discard_before(max(0, scan_sample - pre_roll_samples))

    def transcribe_and_paste_auto(samples, trigger_clicked=False):
        session_state = select_auto_transcriber()
        text = transcribe_sample_block(session_state, samples, 0).strip()
        history_correction = session_state.get("last_transcript_correction")
        if not text:
            print("[auto] no speech detected.")
            print_auto_waiting()
            return
        command_prefix = safe_match_auto_shell_command_prefix(
            text,
            auto_shell_commands,
        )
        if command_prefix is not None:
            shell_command, queued_text = command_prefix
            if auto_trigger_session.get("focus_failed"):
                print(
                    "[auto] focus fallback matched agent name in corrected "
                    "transcript."
                )
            reset_auto_trigger_session(auto_trigger_session)
            if not auto_shell_command_allowed(shell_command, history_correction):
                print_auto_waiting()
                return
            if not run_auto_shell_command(shell_command):
                print_auto_waiting()
                return
            if not queued_text:
                print_auto_waiting()
                return
            paste_started = time.perf_counter()
            PASTE_IN_PROGRESS.set()
            try:
                delay = get_paste_delay()
                if delay:
                    time.sleep(delay)
                if (
                    shell_command.get("tmux_send_target")
                    and get_env_flag("VOICE_AUTO_TMUX_DIRECT_SEND", default=True)
                ):
                    pasted = send_text_to_tmux_target(shell_command, queued_text)
                    paste_action = "sent to tmux and pressed enter"
                else:
                    pasted = type_text_and_submit(queued_text)
                    paste_action = "typed and pressed enter"
            finally:
                PASTE_IN_PROGRESS.clear()
            paste_ms = (time.perf_counter() - paste_started) * 1000
            print(f"[auto] queued text ready: {queued_text}")
            if not pasted:
                report_paste_failure(queued_text)
            else:
                append_transcript_history(
                    queued_text,
                    transcript_history_path,
                    correction=history_correction,
                )
                print(
                    f"[auto] {paste_action} in "
                    f"{paste_ms:.1f} ms: {queued_text}"
                )
            print_auto_waiting()
            return
        shell_command = safe_match_auto_shell_command(text, auto_shell_commands)
        if shell_command is not None:
            reset_auto_trigger_session(auto_trigger_session)
            if not auto_shell_command_allowed(shell_command, history_correction):
                print_auto_waiting()
                return
            if run_auto_shell_command(shell_command):
                print_auto_waiting()
                return
        if is_auto_trigger_session_armed(
            auto_trigger_session,
            auto_trigger_arm_timeout_seconds,
        ):
            queued_text = extract_text_after_trigger_word(
                text,
                auto_trigger_word,
                auto_trigger_aliases,
            )
            if queued_text is None:
                if auto_trigger_session.get("source") == "probe":
                    print(
                        "[auto] ignored utterance: probe heard "
                        f"'{auto_trigger_word}' but final transcript did not."
                    )
                    reset_auto_trigger_session(auto_trigger_session)
                    print_auto_waiting()
                    return
                queued_text = text
        else:
            queued_text = extract_text_after_trigger_word(
                text,
                auto_trigger_word,
                auto_trigger_aliases,
            )
        if queued_text is None:
            print(f"[auto] ignored utterance without '{auto_trigger_word}'.")
            print_auto_waiting()
            return
        if not trigger_clicked and not auto_trigger_session["clicked"]:
            click_auto_trigger_target()
        if auto_trigger_session.get("focus_failed"):
            print(
                "[auto] focus was not confirmed and the corrected transcript "
                "did not start with an agent name; not pasting."
            )
            reset_auto_trigger_session(auto_trigger_session)
            print_auto_waiting()
            return
        if not queued_text:
            if auto_trigger_session.get("source") == "probe":
                print(
                    f"[auto] trigger '{auto_trigger_word}' confirmed with no queued "
                    "text; ending utterance."
                )
                reset_auto_trigger_session(auto_trigger_session)
                print_auto_waiting()
                return
            arm_auto_trigger_session(auto_trigger_session, "trigger_only")
            print(
                f"[auto] trigger '{auto_trigger_word}' armed; "
                "waiting for words after trigger."
            )
            return
        paste_started = time.perf_counter()
        PASTE_IN_PROGRESS.set()
        try:
            delay = get_paste_delay()
            if delay:
                time.sleep(delay)
            pasted = type_text_and_submit(queued_text)
        finally:
            PASTE_IN_PROGRESS.clear()
        paste_ms = (time.perf_counter() - paste_started) * 1000
        if queued_text:
            print(f"[auto] queued text ready: {queued_text}")
        else:
            print("[auto] submit ready.")
        if not pasted:
            report_paste_failure(queued_text)
        else:
            append_transcript_history(
                queued_text,
                transcript_history_path,
                correction=history_correction,
            )
            reset_auto_trigger_session(auto_trigger_session)
            print(f"[auto] typed and pressed enter in {paste_ms:.1f} ms: {queued_text}")
        print_auto_waiting()

    def run_auto_loop():
        start_auto_pause_listener()
        print(
            "[auto] listening in background. Point the mouse at the target "
            "field and start speaking; press Ctrl+C to exit."
        )
        print(
            "[auto] "
            f"start={auto_start_speech_ms:.0f}ms "
            f"vad={'sherpa' if auto_vad_detector is not None else 'rms'} "
            f"trigger='{auto_trigger_word}' "
            f"silence={auto_trigger_silence_seconds:.1f}s "
            f"arm_timeout={auto_trigger_arm_timeout_seconds:.1f}s "
            f"threshold={auto_vad_threshold:.4f}"
        )
        if auto_shell_commands:
            print_auto_waiting()
            labels = sorted(
                {
                    command["label"]
                    for command in auto_shell_commands.values()
                }
            )
            print(
                "[auto] switch words="
                + ", ".join(labels)
            )
        while True:
            if is_auto_paused():
                auto_control["event"].wait()
                auto_control["event"].clear()
                continue
            recording_id = None
            samples = None
            trigger_clicked = False
            try:
                recording_id = recorder.start()
                result = wait_for_auto_utterance()
                if result is not None:
                    samples, trigger_clicked = result
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[auto] recording failed: {exc}", file=sys.stderr)
                time.sleep(1.0)
                continue
            finally:
                recorder.stop(return_samples=False)
                if recording_id is not None:
                    recorder.clear_if_idle(recording_id)

            if samples is None:
                continue
            if samples.size == 0:
                print("[auto] no audio captured.")
                continue
            transcribe_and_paste_auto(samples, trigger_clicked=trigger_clicked)

    if run_mode == "auto":
        try:
            run_auto_loop()
        except KeyboardInterrupt:
            print("\n[auto] stopped.")
        return

    print(
        f"Hold '{hotkey_display}' to record, release to transcribe. "
        "Press Ctrl+C to exit."
    )
    print("[audio] level meter prints while recording.")
    if backend == "evdev":
        try:
            hotkey_code = parse_hotkey_evdev(hotkey_name)
        except Exception as exc:
            print(f"[hotkey] {exc}", file=sys.stderr)
            sys.exit(1)
        listen_hotkey_evdev(
            hotkey_code,
            handle_press,
            handle_release,
            selected_device=evdev_device,
        )
    elif backend == "pynput":
        try:
            hotkey = parse_hotkey_pynput(hotkey_name)
        except Exception as exc:
            print(f"[hotkey] {exc}", file=sys.stderr)
            sys.exit(1)
        listen_hotkey_pynput(hotkey, handle_press, handle_release)
    else:
        print(f"[hotkey] unknown backend '{backend}'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
