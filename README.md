# DeepSeek-OCR-2 on Mac (Apple Silicon)

Running [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) locally on Mac with Apple Silicon GPU acceleration.

## Test Environment

- MacBook Pro with Apple M5 Pro chip, 24GB unified memory
- macOS 15.x (Darwin 25.3.0)
- Python 3.12

> The solutions should work on other Apple Silicon Macs (M1/M2/M3/M4/M5 series). Adjust CPU thread count according to your chip's performance core count.

## TL;DR - Use MLX (Recommended)

```bash
# 1. Create project with uv
uv init ocr2-mlx --python 3.12
cd ocr2-mlx

# 2. Install mlx-vlm
uv add mlx-vlm

# 3. Run OCR
uv run python ocr.py your_image.png
```

The MLX solution runs **entirely on GPU** via Apple's native MLX framework, achieving ~1s inference time.

## Performance Comparison

| Solution | Inference Time | Device | Relative Speed |
|----------|---------------|--------|----------------|
| PyTorch CPU only | ~60s+ | CPU | 1x |
| PyTorch Hybrid (fp32) | 10.10s | CPU + MPS | 6x |
| PyTorch Hybrid (fp16 LLM) | 6.94s | CPU + MPS | 9x |
| **MLX bf16 (full GPU)** | **1.09s** | **Metal GPU** | **55x** |
| Ollama (Q8_0, new engine) | 4.31s cold / **0.77s** warm | Metal GPU (ggml) | 14x / 78x |

## Solution 1: MLX (Full GPU) - Recommended

Uses [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) with pre-converted MLX model from [mlx-community/DeepSeek-OCR-2-bf16](https://huggingface.co/mlx-community/DeepSeek-OCR-2-bf16).

### Setup

```bash
uv init ocr2-mlx --python 3.12
cd ocr2-mlx
uv add mlx-vlm
```

### Usage

See [`mlx/ocr.py`](mlx/ocr.py):

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-2-bf16")

prompt = "Free OCR. "
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model, processor, formatted_prompt,
    ["your_image.png"],
    max_tokens=2048,
    temperature=0.0,
    verbose=False,
)
print(result.text)
```

### Available MLX Models

| Model | Size | Use Case |
|-------|------|----------|
| `mlx-community/DeepSeek-OCR-2-bf16` | ~6.78 GB | Best quality |
| `mlx-community/DeepSeek-OCR-2-8bit` | ~4.03 GB | Good quality, less memory |
| `mlx-community/DeepSeek-OCR-2-4bit` | ~2.5 GB | Smallest, acceptable quality |

### Performance Details (bf16 model)

| Metric | Value |
|--------|-------|
| Model loading (cached) | 1.28s |
| OCR inference | **1.09s** |
| Prompt processing | 1529 tokens/s |
| Token generation | 194 tokens/s |
| Peak memory | 8.46 GB |

---

## Solution 2: PyTorch Hybrid (CPU + MPS)

If you need to use PyTorch/Transformers (e.g., for fine-tuning or integration with existing code), the vision encoder must run on CPU while the language model runs on MPS GPU.

### Setup

```bash
uv init ocr2-pytorch --python 3.12
cd ocr2-pytorch
uv add 'transformers==4.46.3' torch torchvision pillow einops addict easydict accelerate
```

> **Critical**: `transformers` must be `<= 4.47.1`. Version 4.48+ removed `LlamaFlashAttention2` which the model's custom code imports.

### Model Source Code Patching

The original model code from HuggingFace hardcodes `.cuda()` calls and uses `torch.bfloat16`. You must patch the cached model source code before running on Mac.

Apply the patch to `~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR-2/.../modeling_deepseekocr2.py`:

```bash
# After first run downloads the model, apply the patch:
python pytorch/apply_patch.py
```

See [`pytorch/patches/`](pytorch/patches/) for details on what's changed.

### Usage

See [`pytorch/ocr_hybrid.py`](pytorch/ocr_hybrid.py) for the hybrid CPU+MPS solution.

---

## Why Can't PyTorch MPS Run the Full Model?

The vision encoder (DeepEncoderV2 = SAM + Qwen2-0.5B) produces **garbage output** on MPS regardless of dtype (float32, float16, bfloat16). The output looks like `"1. 1. 1. 1. 1. 1."` instead of actual text.

### Root Causes (Multiple PyTorch MPS Bugs)

1. **Non-contiguous tensor silent failures** - SAM's window attention creates non-contiguous tensor views through reshape/permute. MPS silently returns wrong results for operations on these tensors ([PyTorch #165257](https://github.com/pytorch/pytorch/issues/165257))

2. **`masked_scatter_` broken on MPS** - Used to inject vision features into text embeddings. Confirmed to produce garbage data on MPS ([HuggingFace Discussion #20](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20))

3. **SDPA non-contiguous Q/K/V bug** - `scaled_dot_product_attention` with non-contiguous query tensors produces ~34x error magnitude ([PyTorch #163597](https://github.com/pytorch/pytorch/issues/163597))

4. **`F.interpolate` issues** - Bicubic interpolation with antialiasing used for positional embedding resizing was not implemented on MPS before PyTorch 2.7; incorrect results after permute ([PyTorch #88183](https://github.com/pytorch/pytorch/issues/88183))

5. **M-series chip chunk+conv bug** - Tensor views from `chunk()` passed through `Conv2d` produce incorrect results for batch elements > 0 ([PyTorch #169342](https://github.com/pytorch/pytorch/issues/169342))

### Why Ollama/llama.cpp Works on Metal

Ollama uses llama.cpp under the hood, which **re-implements the vision encoder in pure C/ggml** with custom Metal shaders. It does NOT use PyTorch's MPS backend.

### Why MLX Works

[MLX](https://github.com/ml-explore/mlx) is Apple's native ML framework designed specifically for Apple Silicon. All operators have correct Metal implementations — no PyTorch MPS compatibility issues.

---

## Exploration Log

Here's a timeline of what we tried, for anyone going down the same path:

### Attempt 1: PyTorch + Full MPS
- Load model with `AutoModel.from_pretrained()`, move entirely to MPS
- **Result**: Vision encoder outputs garbage regardless of dtype
- **Cause**: Multiple MPS backend bugs (see above)

### Attempt 2: PyTorch + CPU Only
- Run everything on CPU with bfloat16
- **Result**: Correct output, but **~60s+ per image** - too slow

### Attempt 3: PyTorch + Hybrid CPU/MPS (float32)
- Vision encoder (sam_model, qwen2_model, projector) on CPU
- Language model (layers, norm, lm_head) on MPS
- Required patching model source to replace `.cuda()` and add cross-device tensor migration
- **Result**: Correct output, **~10s** per image

### Attempt 4: PyTorch + Hybrid CPU/MPS (float16 LLM)
- Same as above, but LLM layers in float16 on MPS
- **Result**: Correct output, **~7s** per image

### Attempt 5: PyTorch + CPU float16
- Vision encoder in float16 on CPU
- **Result**: **10x slower** than float32! Apple Silicon CPU is optimized for float32, float16 has conversion overhead

### Attempt 6: PyTorch + torch.compile
- Apply `torch.compile(mode="reduce-overhead")` to vision encoder
- **Result**: 110s+ compilation time on first run, not practical for inference

### Attempt 7: MLX via mlx-vlm
- Use pre-converted model `mlx-community/DeepSeek-OCR-2-bf16`
- Full Metal GPU acceleration through Apple's native MLX framework
- **Result**: Correct output, **~1s** per image - **the winner**

### Attempt 8: Ollama (Native ggml/Metal)
- Implement `deepseekocr2` as a new model architecture in Ollama's Go codebase
- Convert HuggingFace safetensors → GGUF, run via Ollama's new engine
- Overcame multiple issues: Metal crashes (cross-context tensors, inconsistent graph shapes), wrong RoPE type causing hallucinated output, hybrid attention mask orientation
- **Result**: Correct output, **4.3s cold / 0.77s warm** - comparable to MLX, integrates with Ollama ecosystem
- See [`ollama/OLLAMA.md`](ollama/OLLAMA.md) for full implementation details and pitfalls

---

## File Structure

```
.
├── README.md
├── mlx/
│   ├── ocr.py                    # MLX solution (recommended)
│   ├── ocr_server.py             # HTTP server for batch OCR
│   └── pyproject.toml
├── pytorch/
│   ├── ocr_cpu.py                # CPU-only fallback
│   ├── ocr_hybrid.py             # Hybrid CPU+MPS solution
│   ├── apply_patch.py            # Auto-patch model source code
│   ├── patches/
│   │   └── PATCHING.md           # What patches are needed and why
│   └── pyproject.toml
├── ollama/
│   ├── OLLAMA.md                 # Implementation guide and pitfalls
│   ├── Modelfile                 # Ollama model template
│   ├── model/deepseekocr2/       # Ollama model implementation (Go)
│   │   ├── model.go              # Main model, EncodeMultimodal, Forward
│   │   ├── model_qwen2.go        # Qwen2 vision encoder
│   │   ├── model_sam.go          # SAM ViT-B encoder
│   │   ├── model_text.go         # DeepSeek MoE text decoder
│   │   └── imageprocessor.go     # Image preprocessing
│   └── convert/
│       └── convert_deepseekocr2.go  # HuggingFace → GGUF converter
└── sample.png                    # Sample test image
```

## License

MIT

## Acknowledgments

- [DeepSeek-AI](https://github.com/deepseek-ai) for DeepSeek-OCR-2
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for MLX model support
- [mlx-community](https://huggingface.co/mlx-community) for pre-converted MLX models
- [Ollama](https://github.com/ollama/ollama) for the inference engine and model framework
