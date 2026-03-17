"""
DeepSeek-OCR-2 Interactive Mode (MLX / Metal GPU)

Loads the model once, then continuously accepts image paths for OCR.

Usage:
    uv run python ocr_server.py
    uv run python ocr_server.py --model mlx-community/DeepSeek-OCR-2-8bit

Controls:
    - Enter image path to run OCR
    - Type 'q', 'quit', 'exit' or press Esc/Ctrl+C to exit
    - Empty input is ignored
"""

import sys
import time
import os
import argparse
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 Interactive OCR")
    parser.add_argument(
        "--model",
        default="mlx-community/DeepSeek-OCR-2-bf16",
        help="MLX model name (default: mlx-community/DeepSeek-OCR-2-bf16)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max tokens to generate"
    )
    args = parser.parse_args()

    # Avoid interactive trust_remote_code prompt
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"

    print(f"Loading model: {args.model} ...")
    t_load = time.time()
    model, processor = load(args.model, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t_load:.2f}s")
    print()
    print("=" * 50)
    print("  DeepSeek-OCR-2 Ready (Metal GPU)")
    print("  Enter image path to run OCR")
    print("  Type 'q' or press Ctrl+C to exit")
    print("=" * 50)

    prompt = "Free OCR. "
    formatted_prompt = apply_chat_template(
        processor, model.config, prompt, num_images=1
    )

    while True:
        try:
            print()
            user_input = input("Image path> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit", "\x1b"):
            print("Bye!")
            break

        # Expand ~ and resolve path
        image_path = os.path.expanduser(user_input)
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        t0 = time.time()
        try:
            result = generate(
                model,
                processor,
                formatted_prompt,
                [image_path],
                max_tokens=args.max_tokens,
                temperature=0.0,
                verbose=False,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        elapsed = time.time() - t0
        text = result.text if hasattr(result, "text") else str(result)

        print(f"\n--- Result ({elapsed:.2f}s) ---")
        print(text)
        print(f"--- {result.generation_tokens} tokens, {result.generation_tps:.0f} tok/s ---")


if __name__ == "__main__":
    main()
