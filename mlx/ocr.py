"""
DeepSeek-OCR-2 on Mac using MLX (Full GPU / Metal acceleration)

Requirements:
    uv init ocr2-mlx --python 3.12
    cd ocr2-mlx && uv add mlx-vlm

Usage:
    uv run python ocr.py image.png
    uv run python ocr.py            # uses sample.png by default
"""

import sys
import time
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template


def ocr(image_path: str, model_name: str = "mlx-community/DeepSeek-OCR-2-bf16"):
    """Run OCR on an image using DeepSeek-OCR-2 via MLX."""
    print(f"Loading model: {model_name} ...")
    t_load = time.time()
    model, processor = load(model_name)
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    prompt = "Free OCR. "
    formatted_prompt = apply_chat_template(
        processor, model.config, prompt, num_images=1
    )

    print(f"Running OCR on: {image_path}")
    t0 = time.time()
    result = generate(
        model,
        processor,
        formatted_prompt,
        [image_path],
        max_tokens=2048,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - t0

    text = result.text if hasattr(result, "text") else str(result)
    print(f"\n===== OCR Result ({elapsed:.2f}s, Metal GPU) =====")
    print(text)
    print("=" * 50)

    # Print performance stats if available
    if hasattr(result, "prompt_tps"):
        print(f"Prompt: {result.prompt_tps:.0f} tokens/s")
        print(f"Generation: {result.generation_tps:.0f} tokens/s")
        print(f"Tokens: {result.prompt_tokens} prompt + {result.generation_tokens} generated")
        print(f"Peak memory: {result.peak_memory:.2f} GB")

    return text


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "../sample.png"
    ocr(image)
