"""
DeepSeek-OCR-2 on Mac using PyTorch (CPU only)

This is the simplest but slowest approach (~60s+ per image).
Works without any model source patching.

Requirements:
    uv init ocr2-pytorch --python 3.12
    cd ocr2-pytorch
    uv add 'transformers==4.46.3' torch torchvision pillow einops addict easydict accelerate

Prerequisites:
    The model source code must be patched first. Run apply_patch.py after the
    first model download.

Usage:
    uv run python ocr_cpu.py image.png
"""

import sys
import time
import torch

# Patch .cuda() -> CPU (model code hardcodes .cuda() calls)
torch.Tensor.cuda = lambda self, *a, **kw: self.to("cpu")

# Patch torch.autocast for non-CUDA device
_original_autocast = torch.autocast
class _autocast_patch:
    def __init__(self, device_type, **kwargs):
        if device_type == "cuda":
            device_type = "cpu"
            kwargs["dtype"] = torch.bfloat16
        self._ctx = _original_autocast(device_type, **kwargs)
    def __enter__(self):
        return self._ctx.__enter__()
    def __exit__(self, *args):
        return self._ctx.__exit__(*args)
torch.autocast = _autocast_patch

from transformers import AutoModel, AutoTokenizer


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "../sample.png"
    model_name = "deepseek-ai/DeepSeek-OCR-2"

    print(f"Loading model: {model_name} ...")
    print("Device: CPU (slow, ~60s+ per image)")

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16).to("cpu")
    print(f"Model loaded in {time.time() - t_load:.2f}s")

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
    elapsed = time.time() - t0

    print(f"\n===== OCR Result ({elapsed:.2f}s, CPU) =====")
    print(res)
    print("=" * 50)


if __name__ == "__main__":
    main()
