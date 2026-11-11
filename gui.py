"""Browser UI for recording or uploading audio and transcribing it.

The model itself stays in the vLLM server started by ./serve.sh.
This process only serves the page and forwards one recording per request.

    .venv/bin/python gui.py
"""

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, OpenAI

from demo_script import VERBATIM, ask, cleanup_generation_loops

SPEAKER_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. "
    "Include all spoken words and verbal fillers. "
    "Write numbers in their standard written form. "
    "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)
SPEAKER_TAG = re.compile(r"<spk:(\d+)>")

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
LIBRARY = ROOT / "recordings"
RECORDING_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
PASSTHROUGH_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

app = FastAPI(title="Nemotron Transcribe")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_speaker_lock = threading.Lock()
_speaker_proc: subprocess.Popen | None = None
_speaker_path: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    return _gpu_stats()


@app.get("/api/recordings")
def recordings():
    return _list_recordings()


@app.get("/api/recordings/{item_id}/audio")
def recording_audio(item_id: str):
    folder = _recording_dir(item_id)
    meta = _read_meta(folder)
    audio = folder / meta["audio"]
    if not audio.is_file():
        raise HTTPException(status_code=404, detail="The recording audio is missing.")
    return FileResponse(audio, media_type=_audio_type(audio.suffix), filename=meta["name"])


@app.delete("/api/recordings/{item_id}")
def delete_recording(item_id: str):
    folder = _recording_dir(item_id)
    shutil.rmtree(folder)
    return {"ok": True}


@app.post("/api/speaker/start")
def speaker_start():
    """Record the default audio output so playback can be mixed into a take."""
    global _speaker_proc, _speaker_path
    with _speaker_lock:
        _stop_speaker_locked()
        try:
            target, description = _default_output()
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc) or "No audio output is available.") from exc
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        proc = subprocess.Popen(
            [
                "pw-record",
                "--target",
                target,
                "--rate",
                "16000",
                "--channels",
                "1",
                "--format",
                "s16",
                tmp.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=0.4)
        except subprocess.TimeoutExpired:
            _speaker_proc = proc
            _speaker_path = tmp.name
            return {"sink": description}
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Speaker output could not be captured.")


@app.post("/api/speaker/stop")
def speaker_stop():
    with _speaker_lock:
        path = _stop_speaker_locked()
    if not path:
        raise HTTPException(status_code=409, detail="Speaker output was not being captured.")
    file = Path(path)
    try:
        if not file.exists() or file.stat().st_size <= 44:
            raise HTTPException(status_code=409, detail="The speaker recording was empty.")
        return Response(content=file.read_bytes(), media_type="audio/wav")
    finally:
        file.unlink(missing_ok=True)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    speakers: str = Form("0"),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The audio file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="The audio file is larger than 80 MB.")

    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in PASSTHROUGH_SUFFIXES:
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        audio_path = Path(tmp.name)
        if suffix == ".wav" and not data.startswith(b"RIFF"):
            audio_path = _convert_to_wav(audio_path)

        client = OpenAI(base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"), api_key="EMPTY")
        speaker_view = speakers.strip().lower() in {"1", "true", "on", "yes"}
        response = ask(
            client,
            audio_path,
            SPEAKER_PROMPT if speaker_view else (prompt.strip() or VERBATIM),
            timestamps=speaker_view,
        )
        text = cleanup_generation_loops(response.choices[0].message.content or "")
        payload = {"text": text}
        if speaker_view:
            payload["turns"] = _speaker_turns(response, text)
        try:
            payload["recording"] = _save_recording(
                audio_path,
                Path(file.filename or "audio.wav").name,
                text,
                payload.get("turns") or [],
            )
        except OSError:
            pass
        return payload
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="The transcription server is not running. Start it with ./serve.sh and try again.",
        ) from exc
    except APIStatusError as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise HTTPException(status_code=exc.status_code, detail=message) from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)
        converted = Path(tmp.name).with_suffix(".converted.wav")
        converted.unlink(missing_ok=True)


def _default_output():
    status = subprocess.check_output(["wpctl", "status"], text=True, timeout=5)
    sink_id = None
    description = ""
    in_sinks = False
    for line in status.splitlines():
        if "Sinks:" in line:
            in_sinks = True
            continue
        if in_sinks and "├─" in line:
            break
        if not in_sinks:
            continue
        match = re.search(r"\*\s+(\d+)\.\s+(.+?)\s+\[", line)
        if match:
            sink_id = match.group(1)
            description = match.group(2).strip()
            break
    if not sink_id:
        raise RuntimeError("No audio output is available.")
    inspect = subprocess.check_output(["wpctl", "inspect", sink_id], text=True, timeout=5)
    name = re.search(r'node\.name = "([^"]+)"', inspect)
    return (name.group(1) if name else sink_id), description


def _stop_speaker_locked():
    global _speaker_proc, _speaker_path
    proc = _speaker_proc
    path = _speaker_path
    _speaker_proc = None
    _speaker_path = None
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    return path


def _audio_type(suffix: str) -> str:
    return {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(suffix.lower(), "application/octet-stream")


def _recording_dir(item_id: str) -> Path:
    if not RECORDING_ID.fullmatch(item_id):
        raise HTTPException(status_code=404, detail="Recording not found.")
    folder = (LIBRARY / item_id).resolve()
    if folder.parent != LIBRARY.resolve() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Recording not found.")
    return folder


def _read_meta(folder: Path) -> dict:
    try:
        meta = json.loads((folder / "meta.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Recording not found.") from exc
    if not isinstance(meta, dict) or not meta.get("audio"):
        raise HTTPException(status_code=404, detail="Recording not found.")
    return meta


def _save_recording(audio_path: Path, filename: str, text: str, turns: list) -> dict:
    LIBRARY.mkdir(exist_ok=True)
    item_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = LIBRARY / item_id
    folder.mkdir()
    stored = "audio" + (audio_path.suffix.lower() or ".wav")
    shutil.copyfile(audio_path, folder / stored)
    meta = {
        "id": item_id,
        "name": Path(filename).name or stored,
        "created": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "turns": turns,
        "audio": stored,
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    return meta


def _list_recordings() -> list:
    if not LIBRARY.is_dir():
        return []
    items = []
    for folder in LIBRARY.iterdir():
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict) and meta.get("id"):
            items.append(meta)
    items.sort(key=lambda item: item.get("created") or "", reverse=True)
    return items


def _number(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_stats():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).strip().splitlines()[0]
    name, util, memory_util, power, temp = [part.strip() for part in raw.split(",")]
    meminfo = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0])
    total_kb = meminfo.get("MemTotal", 0)
    available_kb = meminfo.get("MemAvailable", 0)
    return {
        "name": name,
        "utilization": _number(util),
        "memory_utilization": _number(memory_util),
        "power_watts": _number(power),
        "temperature_c": _number(temp),
        "memory_total_gb": total_kb / 1024 / 1024,
        "memory_used_gb": (total_kb - available_kb) / 1024 / 1024,
    }


def _speaker_turns(response, text: str):
    alignment = response.model_dump().get("ctc_timestamps") or {}
    words = alignment.get("words") or []
    if words and alignment.get("error") is None:
        turns = []
        for word in words:
            speaker = word.get("speaker")
            piece = (word.get("word") or "").strip()
            if not piece:
                continue
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["text"] = f"{turns[-1]['text']} {piece}"
                turns[-1]["end"] = word.get("end")
            else:
                turns.append(
                    {
                        "speaker": speaker,
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "text": piece,
                    }
                )
        if turns:
            return turns
    return _turns_from_tags(text)


def _turns_from_tags(text: str):
    parts = SPEAKER_TAG.split(text)
    if len(parts) == 1:
        spoken = text.strip()
        return [{"speaker": None, "start": None, "end": None, "text": spoken}] if spoken else []
    turns = []
    lead = parts[0].strip()
    if lead:
        turns.append({"speaker": None, "start": None, "end": None, "text": lead})
    for speaker, content in zip(parts[1::2], parts[2::2]):
        spoken = content.strip()
        if spoken:
            turns.append({"speaker": int(speaker), "start": None, "end": None, "text": spoken})
    return turns


def _convert_to_wav(src: Path) -> Path:
    """Decode a non-wav upload, such as a browser recording stored without a header, to 16 kHz mono wav."""
    import av

    dst = src.with_suffix(".converted.wav")
    with av.open(str(src)) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        with av.open(str(dst), mode="w", format="wav") as output:
            out_stream = output.add_stream("pcm_s16le", rate=16000, layout="mono")
            for frame in container.decode(stream):
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                frames = resampled if isinstance(resampled, list) else [resampled]
                for item in frames:
                    item.pts = None
                    for packet in out_stream.encode(item):
                        output.mux(packet)
            for packet in out_stream.encode(None):
                output.mux(packet)
    src.unlink(missing_ok=True)
    return dst


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GUI_PORT", "8080")))
