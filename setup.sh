#!/usr/bin/env bash
# Install the NVFP4 runtime from the NeMo Speech lockfile, then download the checkpoint.
# See https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NEMO="${ROOT}/nemo-speech"
VENV="${ROOT}/.venv"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
MODEL_DIR="${ROOT}/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
SAMPLE_URL="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
SAMPLE="${ROOT}/samples/en-Alice_woman.wav"

if [[ ! -f "${NEMO}/uv.lock" ]]; then
  git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git "${NEMO}"
fi

cd "${NEMO}"
# NeMo Speech pins Python 3.12 in .python-version. vLLM 0.32.0 accepts 3.11–3.14.
# Passing --python keeps uv sync from recreating the environment with another interpreter.
uv venv --python 3.12 "${VENV}"
UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --python 3.12 --extra asr --extra vllm --locked --no-install-project
uv pip install --python "${VENV}/bin/python" --no-deps -e .

# Install these from outside nemo-speech. Its override-dependencies marker
# drops opencv-python-headless, and vLLM imports cv2 at startup.
cd "${ROOT}"
# PyAV decodes compressed audio such as M4A without a system FFmpeg.
uv pip install --python "${VENV}/bin/python" --no-deps av==19.0.2
uv pip install --python "${VENV}/bin/python" --no-deps opencv-python-headless==5.0.0.93

mkdir -p "${ROOT}/models" "${ROOT}/samples"
# The shared xet cache under ~/.cache/huggingface/xet is not writable here.
export HF_HUB_DISABLE_XET=1
"${VENV}/bin/hf" download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
if [[ ! -f "${SAMPLE}" ]]; then
  curl -fsSL "${SAMPLE_URL}" -o "${SAMPLE}"
fi

echo "Ready. Start the server with ./serve.sh, then run ./demo_script.py"
