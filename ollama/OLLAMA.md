# DeepSeek-OCR-2 on Ollama (Native Metal)

Running DeepSeek-OCR-2 natively on Ollama with the new engine, entirely on Metal GPU via ggml.

> **Status**: Working prototype. Tested on Ollama commit `bbbad97`. Not yet merged upstream.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Go 1.24+ (tested with Go 1.24.3)
- Ollama source at commit `bbbad97` (see below)
- ~8 GB free memory

## Quick Start (Pre-built Binary + GGUF from Release)

**This is the recommended way.** Download from [GitHub Release](https://github.com/larryteal/Mac-M5-Deepseek-OCR-2/releases/tag/v0.2.0-ollama):

```bash
# 1. Download all files from the Release page, then:
chmod +x ollama-test-darwin-arm64

# 2. Merge model parts into one GGUF file
cat deepseek-ocr2-q8_0.gguf.part_a* > deepseek-ocr2-q8_0.gguf

# 3. Verify checksums (optional)
shasum -a 256 deepseek-ocr2-q8_0.gguf.part_a* | diff - SHA256SUMS.txt

# 4. Create Ollama model from GGUF
OLLAMA_NEW_ENGINE=1 GGML_METAL_TENSOR_DISABLE=1 ./ollama-test-darwin-arm64 serve &
sleep 3

cat > Modelfile <<'EOF'
FROM ./deepseek-ocr2-q8_0.gguf
TEMPLATE """<｜begin▁of▁sentence｜><|User|>{{ if .Images }}<image>
{{ end }}{{ .Prompt }}<|Assistant|>"""
PARAMETER stop <｜end▁of▁sentence｜>
PARAMETER stop <|User|>
PARAMETER temperature 0
EOF

./ollama-test-darwin-arm64 create deepseek-ocr2 -f Modelfile

# 5. Run OCR
./ollama-test-darwin-arm64 run deepseek-ocr2 "Free OCR. /path/to/image.png"
```

> **Port conflict?** If you already have Ollama running, add `OLLAMA_HOST=127.0.0.1:11447` before every command to use a different port.

## Build from Source

If you want to build yourself instead of using the pre-built binary:

### Step 1: Clone Ollama at the tested commit

```bash
git clone https://github.com/ollama/ollama.git
cd ollama
git checkout bbbad97
```

> **Important**: This code was tested on commit `bbbad97`. Other versions may have incompatible internal API changes.

### Step 2: Copy model files from this repo

```bash
# Assuming this repo is cloned to ../Mac-M5-Deepseek-OCR-2
REPO=../Mac-M5-Deepseek-OCR-2/ollama

# Model implementation
mkdir -p model/models/deepseekocr2
cp ${REPO}/model/deepseekocr2/*.go model/models/deepseekocr2/

# GGUF converter
cp ${REPO}/convert/convert_deepseekocr2.go convert/
```

### Step 3: Apply patches to Ollama source

Apply all patches with one command:

```bash
git apply ${REPO}/ollama-deepseekocr2.patch
```

If you prefer to apply manually, four files need one-line edits each:

**`model/models/models.go`** — register the model architecture (add one import line):
```diff
  _ "github.com/ollama/ollama/model/models/deepseekocr"
+ _ "github.com/ollama/ollama/model/models/deepseekocr2"
  _ "github.com/ollama/ollama/model/models/gemma2"
```

**`convert/convert.go`** — register the GGUF converter (add one case in `LoadModelMetadata`):
```diff
  case "DeepseekOCRForCausalLM":
      conv = &deepseekocr{}
+ case "DeepseekOCR2ForCausalLM":
+     conv = &deepseekocr2{}
  case "DeepseekV3ForCausalLM":
```

**`ml/backend/ggml/ggml.go`** — route Qwen2 vision tensors (`q.*`) to the output context instead of the layer context (prevents index-out-of-range panic):
```diff
- case strings.HasPrefix(t.Name, "v.") || strings.HasPrefix(t.Name, "mm.") || strings.HasPrefix(t.Name, "s."):
+ case strings.HasPrefix(t.Name, "v.") || strings.HasPrefix(t.Name, "mm.") || strings.HasPrefix(t.Name, "s.") || strings.HasPrefix(t.Name, "q."):
      // TODO: assign vision tensors to the gpu if possible
      createTensor(tensor{source: t}, output.bts, blocks)
```

**`server/images.go`** — enable vision capability detection for CLI image support:
```diff
- if f.KeyValue("vision.block_count").Valid() {
+ if f.KeyValue("vision.block_count").Valid() || f.KeyValue("qwen2.block_count").Valid() {
      capabilities = append(capabilities, model.CapabilityVision)
```

### Step 4: Build

```bash
CGO_ENABLED=1 go build -o ollama-test .
```

### Step 5: Create model and run

Use the GGUF from the Release, or convert from HuggingFace yourself:

```bash
# Option A: Use GGUF from Release (recommended)
# Download and merge parts as shown in Quick Start above, then:
OLLAMA_NEW_ENGINE=1 GGML_METAL_TENSOR_DISABLE=1 ./ollama-test serve &
sleep 3
./ollama-test create deepseek-ocr2 -f Modelfile

# Option B: Convert from HuggingFace (requires local safetensors)
# First download the model: huggingface-cli download deepseek-ai/DeepSeek-OCR-2
# Then create Modelfile with: FROM /path/to/hf/snapshot
# And run: ./ollama-test create deepseek-ocr2 -f Modelfile --experimental
# Note: Q8_0 quantization requires MLX. Without -q flag, F16 is created (~12GB).
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_NEW_ENGINE=1` | **Yes** | Enable the new Ollama engine (required for deepseekocr2) |
| `GGML_METAL_TENSOR_DISABLE=1` | **Yes** | Avoid Metal residency set issues with new engine |
| `OLLAMA_HOST=127.0.0.1:PORT` | No | Use a different port if default 11434 is occupied |

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

## Key Pitfalls (Lessons Learned)

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

**Fix**: Patch `server/images.go` to also check `qwen2.block_count` (see Step 3 in Build from Source above).

### Pitfall 7: Tensor Name Mapping Order

**Symptom**: Qwen2 tensors not loaded; model fails validation.

**Root Cause**: In `convert_deepseekocr2.go`, the replacement rule `"model.norm" -> "output_norm"` matches inside `"model.qwen2_model.model.model.norm"`, preventing the Qwen2 output norm from being mapped correctly.

**Fix**: Place more specific replacements before general ones:
```go
// MUST come before "model.norm" -> "output_norm"
"model.qwen2_model.model.model.norm", "q.output_norm",
```

### Pitfall 8: Qwen2 Tensor Routing Causes Panic on Load

**Symptom**: `runtime error: index out of range [12] with length 12` when loading the model.

**Root Cause**: Ollama's ggml backend routes tensors to GPU layers by extracting the first number from tensor names (e.g., `blk.5.attn_q` -> layer 5). Qwen2 tensors like `q.blk.12.attn_q` extract `12`, but the text model only has 12 blocks (index 0-11), causing an out-of-bounds access.

**Fix**: Patch `ml/backend/ggml/ggml.go` to route `q.*` tensors (Qwen2 vision encoder) to the output context alongside other vision tensors (`v.*`, `mm.*`, `s.*`). See Step 3 in Build from Source above.

## Source Files

```
ollama/
├── OLLAMA.md                          # This file
├── Modelfile                          # Template for creating the model
├── ollama-deepseekocr2.patch          # All Ollama source patches (git apply)
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

Tested on MacBook Pro M5 Pro 24GB, macOS 15.x, Go 1.24.3, Ollama commit `bbbad97`:

| Metric | Cold Start | Warm (model loaded) |
|--------|-----------|-------------------|
| Total time | **4.31s** | **0.77s** |
| Model loading | 1.20s | 0.05s |
| Prompt eval (697 tokens) | 2.40s (291 tok/s) | 0.02s (cached) |
| Text generation (109 tokens) | 0.59s (184 tok/s) | 0.60s (183 tok/s) |
| Model size (Q8_0) | ~5.6 GB | |
| Memory usage | ~8 GB | |

## Comparison with Other Solutions

| Solution | Inference | Quality | Setup Complexity |
|----------|-----------|---------|-----------------|
| MLX bf16 | ~1.09s | Best (bf16) | Low (pip install) |
| PyTorch Hybrid | ~7s | Best (fp32/fp16) | Medium (patching needed) |
| **Ollama (this)** | **0.77s warm / 4.3s cold** | **Good (Q8_0)** | **Low (pre-built) / High (from source)** |

Ollama warm inference is comparable to MLX. Cold start includes model loading (~1.2s) and first-time prompt encoding (~2.4s). The main advantage is integration with the Ollama ecosystem (API, CLI, model management, OpenAI-compatible endpoints).
