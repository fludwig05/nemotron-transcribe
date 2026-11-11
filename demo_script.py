"""Transcribe a recording with NVIDIA Nemotron 3.5 Transcribe NVFP4.

Start the server first:

    ./serve.sh

Then, in another shell:

    .venv/bin/python demo_script.py
    .venv/bin/python demo_script.py path/to/audio.wav
"""

import argparse
import base64
import re
import sys
import urllib.request
from pathlib import Path

from openai import OpenAI

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT / "samples" / "en-Alice_woman.wav"
SAMPLE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples"
    "/resolve/main/en-Alice_woman.wav"
)
VERBATIM = (
    "Audio transcription. Language: as spoken. "
    "Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. "
    "Output: transcript only."
)

# Collapse rare generation loops before returning a transcript.
# Adapted from the Qwen3-ASR repetition fix (Apache-2.0).
CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda match: match.group(1), text)
    return BLOCK_RUN.sub(lambda match: match.group(1), text)


def ensure_sample(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample audio to {path}", file=sys.stderr)
    urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def mime_for(path: Path) -> str:
    return {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(path.suffix.lower(), "audio/wav")


def ask(client, audio_path, prompt, system_prompt="", reasoning=False, timestamps=False, max_tokens=None):
    audio = base64.b64encode(Path(audio_path).read_bytes()).decode()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:{mime_for(Path(audio_path))};base64,{audio}"},
                },
            ],
        }
    )
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": reasoning},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "seed": 0,
    }
    # serve.sh starts vLLM with max_model_len=8192, and max_tokens cannot exceed that.
    if max_tokens is None:
        max_tokens = 4096 if reasoning else 2048
    if reasoning:
        extra_body.update(top_k=-1, thinking_token_budget=max_tokens)
        extra_body["chat_template_kwargs"]["reasoning_budget"] = max_tokens - 1024
    if timestamps:
        extra_body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.6 if reasoning else 0,
        top_p=0.95 if reasoning else 1.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Audio file. Defaults to a short English sample.",
    )
    parser.add_argument("--prompt", default=VERBATIM, help="Instruction sent with the audio.")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt.")
    parser.add_argument("--reasoning", action="store_true", help="Enable thinking. Use for questions, not transcription.")
    parser.add_argument("--timestamps", action="store_true", help="Request word timestamps and print them.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    audio = ensure_sample(args.audio or DEFAULT_AUDIO)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    response = ask(
        client,
        audio,
        args.prompt,
        system_prompt=args.system_prompt,
        reasoning=args.reasoning,
        timestamps=args.timestamps,
    )
    text = cleanup_generation_loops(response.choices[0].message.content or "")
    print(text)

    if args.timestamps:
        alignment = response.model_dump().get("ctc_timestamps")
        if not alignment:
            print("No ctc_timestamps on the response.", file=sys.stderr)
            return
        if alignment.get("error"):
            raise SystemExit(alignment["error"])
        for word in alignment["words"]:
            print(f"{word['start']:.2f}-{word['end']:.2f} [{word['speaker']}] {word['word']}")


if __name__ == "__main__":
    main()
