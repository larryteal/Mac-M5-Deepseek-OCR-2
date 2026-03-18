# DeepSeek-OCR-2 on Ollama (Native Metal)

Running DeepSeek-OCR-2 natively on Ollama with the new engine (`OLLAMA_NEW_ENGINE=1`).

This is an experimental implementation that adds `deepseekocr2` as a new model architecture to Ollama. It runs entirely on Metal GPU via ggml, achieving ~57s total inference (model load + OCR) with correct output.

> **Status**: Working prototype. Not yet merged upstream into Ollama.

## Architecture

DeepSeek-OCR-2 uses a unique 3-stage vision encoder (different from typical CLIP-based models):

```
Image → SAM ViT-B → Qwen2-0.5B (decoder-as-encoder) → Projector → LLM (DeepSeek MoE)
         │                  │                              │
    Patch features   Hybrid attention          Project to LLM dim
    [H,W,C,batch]   (bidirectional for        [896] → [1280]
                     image, causal for
                     query tokens)
```

- **SAM ViT-B**: Segment Anything Model extracts spatial features from image patches
- **Qwen2-0.5B**: A 24-layer Qwen2 transformer used as encoder (not decoder), with learned query embeddings and hybrid attention mask
- **Projector**: Linear projection from Qwen2 hidden dim (896) to LLM hidden dim (1280)
- **LLM**: DeepSeek MoE language model (same architecture as DeepSeek-V2-Lite)

## Quick Start

### 1. Build Ollama from Source

Clone Ollama and add the deepseekocr2 model files:

```bash
git clone https://github.com/ollama/ollama.git
cd ollama

# Copy model implementation
mkdir -p model/models/deepseekocr2
cp <this-repo>/ollama/model/deepseekocr2/*.go model/models/deepseekocr2/

# Copy GGUF converter
cp <this-repo>/ollama/convert/convert_deepseekocr2.go convert/

# Register the model architecture (add import to model/models/models.go):
#   _ "github.com/ollama/ollama/model/models/deepseekocr2"

# Register the converter (add case to convert/converter.go):
#   case "DeepseekOCR2ForCausalLM":
#       return &deepseekocr2{}, nil

# Build
CGO_ENABLED=1 go build -o ollama-test .
```

### 2. Convert and Create Model

```bash
# Convert from HuggingFace (downloads ~6GB)
./ollama-test create deepseek-ocr2 -q Q8_0 https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
```

### 3. Start Server and Run

```bash
# Start Ollama with new engine
OLLAMA_NEW_ENGINE=1 ./ollama-test serve

# In another terminal:
./ollama-test run deepseek-ocr2 "Free OCR. /path/to/image.png"
```

## Key Implementation Decisions and Pitfalls

### Pitfall 1: RoPE Type (Critical - Causes Hallucinated Output)

**Symptom**: Model detects bounding boxes correctly but outputs hallucinated text instead of actual OCR content.

**Root Cause**: Qwen2 uses **NeoX-style RoPE** (`rotate_half`), but the default in Ollama/ggml is GPT-J style. This completely corrupts positional encoding across all 24 Qwen2 transformer blocks.

**Fix**: Add `rope.WithTypeNeoX()` to RoPE calls:
```go
// model_qwen2.go - WRONG (default GPT-J style):
query = nn.RoPE(ctx, query, positions, opts.headDim(), opts.ropeTheta, 1.0)

// CORRECT (NeoX style, matching HuggingFace Qwen2):
query = nn.RoPE(ctx, query, positions, opts.headDim(), opts.ropeTheta, 1.0, rope.WithTypeNeoX())
```

**How to verify**: Check `transformers/models/qwen2/modeling_qwen2.py` - it defines `rotate_half()` and uses it in `apply_rotary_pos_emb()`. Also, Ollama's own `model/models/qwen2/model.go` uses `rope.WithTypeNeoX()`.

### Pitfall 2: Metal Crash from Cross-Context Tensor Access

**Symptom**: `MTLIOAccelCommandBuffer.setResidencySet` assertion failure on the second image.

**Root Cause**: Model weight tensors (Query768/Query1024) live in the model-loading context. Using them directly in the EncodeMultimodal compute graph creates cross-context Metal residency set conflicts.

**Fix**: Copy query weight data into the current compute context:
```go
queryData := srcQuery.Floats()
if len(queryData) > 0 {
    queryEmbed = ctx.Input().FromFloats(queryData, srcQuery.Dim(0), nQueries)
} else {
    queryEmbed = srcQuery.Cast(ctx, ml.DTypeF32)
}
```

### Pitfall 3: Metal Crash from Inconsistent Tensor Shapes

**Symptom**: Crash on images with different aspect ratios than the probe image.

**Root Cause**: The ggml Metal backend requires identical tensor shapes between the probe (model loading) and actual inference. Different images produce different numbers of patches (2-6 depending on aspect ratio).

**Fix**: Always pad to `maxNum` (6) patches, then slice to keep only real outputs:
```go
// Pad patches to fixed count
for blocks < maxNum {
    patches = append(patches, make([]float32, patchSize)...)
    blocks++
}

// After processing, slice to real tokens only
realTokens := nQueriesLocal * realBlocks
localOutputs = localOutputs.Slice(ctx, 1, 0, realTokens, 1)
```

### Pitfall 4: Image Normalization Mismatch

**Symptom**: Poor OCR quality, model seems confused by image content.

**Root Cause**: DeepSeek-OCR-2 uses `[0.5, 0.5, 0.5]` normalization (maps pixels to [-1, 1]), NOT the ImageNet standard `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`. Check `processor_config.json` in the HuggingFace repo.

### Pitfall 5: Hybrid Attention Mask

**Symptom**: Garbled output even with correct RoPE.

**Root Cause**: The Qwen2 encoder uses a hybrid attention mask - bidirectional for image tokens, causal for query tokens (with cross-attention to all image tokens). Standard causal masking breaks the image understanding.

The mask layout in ggml is `[key, query]` (column-major), which is transposed compared to the intuitive `[query, key]` layout. Getting this wrong silently produces garbage.

### Pitfall 6: Vision Capability Not Detected by CLI

**Symptom**: `ollama run` ignores image paths in the prompt.

**Root Cause**: Ollama CLI checks for `vision.block_count` in GGUF metadata to decide if a model supports images. Our model uses `qwen2.block_count` and `sam.block_count` instead.

**Fix**: Add `vision.block_count` in the GGUF converter, or extend the capability detection in `server/images.go`.

### Pitfall 7: Tensor Name Mapping Order

**Symptom**: Qwen2 tensors not loaded; model fails validation.

**Root Cause**: In `convert_deepseekocr2.go`, the replacement rule `"model.norm" → "output_norm"` matches inside `"model.qwen2_model.model.model.norm"`, preventing the Qwen2 output norm from being mapped correctly.

**Fix**: Place more specific replacements before general ones:
```go
// MUST come before "model.norm" → "output_norm"
"model.qwen2_model.model.model.norm", "q.output_norm",
```

## Source Files

```
ollama/
├── OLLAMA.md                          # This file
├── Modelfile                          # Template for creating the model
├── model/deepseekocr2/
│   ├── model.go                       # Main model, EncodeMultimodal, Forward
│   ├── model_qwen2.go                 # Qwen2 vision encoder (24 blocks)
│   ├── model_sam.go                   # SAM ViT-B encoder
│   ├── model_text.go                  # DeepSeek MoE text decoder
│   └── imageprocessor.go             # Image preprocessing, patch extraction
└── convert/
    └── convert_deepseekocr2.go        # HuggingFace → GGUF converter
```

## Performance

Tested on MacBook Pro M5 Pro 24GB:

| Metric | Value |
|--------|-------|
| Model size (Q8_0) | ~5.6 GB |
| First inference (cold) | ~57s |
| Subsequent inference (warm) | ~45s |
| Memory usage | ~8 GB |

> Note: The new engine (`OLLAMA_NEW_ENGINE=1`) is required. Performance will likely improve as the new engine matures.

## Comparison with Other Solutions

| Solution | Inference | Quality | Setup Complexity |
|----------|-----------|---------|-----------------|
| MLX bf16 | ~1s | Best | Low (pip install) |
| PyTorch Hybrid | ~7s | Best | Medium (patching needed) |
| **Ollama (this)** | **~45s** | **Good** | **High (build from source)** |

Ollama's advantage is integration with the Ollama ecosystem (API, CLI, model management). For raw performance, MLX is significantly faster on Apple Silicon.
