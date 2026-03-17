"""
DeepSeek-OCR-2 on Mac using PyTorch (Hybrid: CPU vision + MPS LLM)

Vision encoder runs on CPU (MPS produces garbage output due to PyTorch bugs).
Language model runs on MPS (Metal GPU) with float16 for speed.

This achieves ~7s per image - much faster than CPU-only (~60s+),
but still slower than the MLX solution (~1s).

Requirements:
    uv init ocr2-pytorch --python 3.12
    cd ocr2-pytorch
    uv add 'transformers==4.46.3' torch torchvision pillow einops addict easydict accelerate

Prerequisites:
    The model source code must be patched first. See patches/PATCHING.md
    or run: python apply_patch.py

Usage:
    uv run python ocr_hybrid.py image.png
"""

import sys
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "../sample.png"
    model_name = "deepseek-ai/DeepSeek-OCR-2"
    device = "mps"

    # Set thread count to performance core count for optimal CPU throughput.
    # Adjust this for your chip:
    #   M1/M2/M3: 8, M1 Pro/Max: 8, M2 Pro: 8, M3 Pro: 6,
    #   M4: 4, M4 Pro: 10, M5 Pro: 10
    perf_cores = 10
    torch.set_num_threads(perf_cores)

    print(f"Loading model: {model_name} ...")
    print(f"CPU threads: {torch.get_num_threads()}, LLM device: {device}")

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    model = model.eval()

    # Set device routing attributes (used by patched model code)
    model._target_device = "cpu"
    model.model._target_device = "cpu"
    model.model._vision_device = "cpu"
    model.model._llm_device = device

    # Vision encoder stays on CPU (MPS produces garbage output)
    model.model.sam_model.to("cpu")
    model.model.qwen2_model.to("cpu")
    model.model.projector.to("cpu")
    model.model.embed_tokens.to("cpu")

    # Language model goes to MPS with float16 (faster + less memory)
    model.model.layers.to(torch.float16).to(device)
    model.model.norm.to(torch.float16).to(device)
    model.lm_head.to(torch.float16).to(device)

    if hasattr(model.model, "view_seperator"):
        model.model.view_seperator = torch.nn.Parameter(
            model.model.view_seperator.data.to("cpu"), requires_grad=False
        )

    print(f"Model loaded in {time.time() - t_load:.2f}s")
    print("Vision encoder -> CPU (float32), Language model -> MPS (float16)")

    # Optional: timing hooks for profiling
    _orig_sam = model.model.sam_model.forward
    _orig_qwen2 = model.model.qwen2_model.forward
    timings = {"sam": 0, "qwen2": 0}

    def timed_sam(*a, **kw):
        t = time.time()
        r = _orig_sam(*a, **kw)
        timings["sam"] += time.time() - t
        return r

    def timed_qwen2(*a, **kw):
        t = time.time()
        r = _orig_qwen2(*a, **kw)
        timings["qwen2"] += time.time() - t
        return r

    model.model.sam_model.forward = timed_sam
    model.model.qwen2_model.forward = timed_qwen2

    print(f"Running OCR on: {image_path}")

    t0 = time.time()
    res = model.infer(
        tokenizer,
        prompt="<image>\nFree OCR. ",
        image_file=image_path,
        output_path="/tmp/ocr2_output",
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=False,
        eval_mode=True,
    )
    total = time.time() - t0

    vision_total = timings["sam"] + timings["qwen2"]
    llm_time = total - vision_total

    print(f"\n===== OCR Result ({total:.2f}s) =====")
    print(res)
    print("=" * 50)
    print(f"Vision encoder (CPU):  {vision_total:.2f}s  (sam: {timings['sam']:.2f}s, qwen2: {timings['qwen2']:.2f}s)")
    print(f"Language model (MPS):  {llm_time:.2f}s")


if __name__ == "__main__":
    main()
