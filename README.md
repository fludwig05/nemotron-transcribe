# Nemotron 3.5 Transcribe NVFP4

Demo for [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) on this DGX Spark (GB10).

The checkpoint is a ModelOpt NVFP4/FP8 export. It runs through vLLM 0.32.0 and the NeMo Speech plugin. `setup.sh` builds that stack with uv from NeMo Speech's lockfile, into `.venv`, using Python 3.12.

```bash
./setup.sh          # once: install the stack and download the checkpoint
./serve.sh          # terminal 1; first start compiles FlashInfer kernels
.venv/bin/python demo_script.py
.venv/bin/python demo_script.py path/to/recording.wav --timestamps
.venv/bin/python gui.py     # http://127.0.0.1:8080 — record or upload
```

`serve.sh` keeps this Spark's shared RAM intact. vLLM gets 45% of it, recordings are capped at 10 minutes, and FlashInfer compiles one kernel at a time before the weights load. The model card's profile (85% of a 96 GB discrete GPU, one-hour audio) is what rebooted this machine.

Transcription uses temperature 0 and reasoning off. Ask a question with `--reasoning` and a question as `--prompt`.
