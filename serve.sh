#!/usr/bin/env bash
# Serve the NVFP4 checkpoint with the profile from the model card.
# FlashInfer compiles quantized kernels on the first start, which needs nvcc and takes several minutes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/.venv"
MODEL="${MODEL_DIR:-${ROOT}/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4}"

if [[ ! -x "${VENV}/bin/vllm" ]]; then
  echo "vLLM is not installed. Run ./setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Checkpoint not found at ${MODEL}. Run ./setup.sh first." >&2
  exit 1
fi

# The GB10 shares one memory pool between CPU and GPU. A CUDA allocation that
# does not fit is handled by the kernel OOM killer, which takes down the session.
available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
available_gb="$(awk -v kb="${available_kb}" 'BEGIN {printf "%.0f", kb/1024/1024}')"
min_gb="${MIN_FREE_GB:-70}"
if (( available_gb < min_gb )); then
  echo "Refusing to load the model: only ${available_gb} GB of RAM is free (need ${min_gb} GB)." >&2
  echo "On this Spark the GPU shares system RAM. Free memory, then run ./serve.sh again." >&2
  exit 1
fi

# FlashInfer's JIT invokes `ninja` by name. The binary lives in the venv,
# which is not on PATH when this script is started without activating it.
export PATH="${VENV}/bin:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_PLUGINS=nemo_speechlm
# One nvcc at a time. Parallel cicc processes were what ran the box out of RAM.
export MAX_JOBS=1
export FLASHINFER_NVCC_THREADS=1
# Word timestamps require the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0
# Keep the CUTLASS FP8 kernels from the validated NVFP4 profile.
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
# Reuse compiled FlashInfer kernels across restarts.
export FLASHINFER_WORKSPACE_BASE="${ROOT}/.flashinfer"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

# 10 minutes at 16 kHz. The model card's one-hour profile makes vLLM reserve
# an hour of audio during startup, which does not fit alongside the weights.
audio_seconds="${AUDIO_SECONDS:-600}"
audio_samples="$((audio_seconds * 16000))"
export VLLM_MAX_AUDIO_DECODE_DURATION_S="${audio_seconds}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180

# Compile the sampling kernel before the weights are loaded. The previous
# start died here, with the model already resident and several compilers running.
if ! find "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" \
  -path '*/cached_ops/sampling/sampling.so' -print -quit 2>/dev/null | grep -q .; then
  echo "Compiling FlashInfer sampling kernels before loading the model..."
  "${VENV}/bin/python" -c "from flashinfer.jit.sampling import gen_sampling_module; gen_sampling_module().build_and_load()"
fi

exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.45}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-batched-tokens 4096 \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --use-replayssm \
  --mamba-backend flashinfer \
  --mamba-cache-mode none \
  --mm-processor-cache-gb 0 \
  --no-async-scheduling \
  --limit-mm-per-prompt "{\"audio\":{\"count\":1,\"length\":${audio_samples}}}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
  --host 127.0.0.1 \
  --port "${PORT:-8000}"
