package deepseekocr2

import (
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

// Qwen2 decoder-as-encoder: replaces CLIP in OCR v1.
// Uses hybrid attention: non-causal for image tokens, causal for query tokens.
//
// All tensors in ggml follow [Dim0=innermost, Dim1, Dim2=batch] convention.
// nn.Linear operates on Dim(0), so hidden dimension must be Dim(0).
// Standard layout: [hiddenDim, seqLen, batch]

type qwen2Model struct {
	Blocks     []qwen2Block `gguf:"blk"`
	OutputNorm *nn.RMSNorm  `gguf:"output_norm"`
	Query768   ml.Tensor    `gguf:"query_768.weight"`
	Query1024  ml.Tensor    `gguf:"query_1024.weight"`

	Options qwen2Options
}

type qwen2Options struct {
	hiddenSize       int
	numHeads         int
	numKVHeads       int
	intermediateSize int
	eps              float32
	ropeTheta        float32
}

func (o qwen2Options) headDim() int {
	return o.hiddenSize / o.numHeads
}

// Forward takes SAM output features and runs them through the Qwen2 encoder
// with learned query embeddings using hybrid attention.
//
// SAM outputs [H, W, C, batch] in ggml. We need [C, seqLen, batch] for transformer.
func (m *qwen2Model) Forward(ctx ml.Context, samFeatures ml.Tensor) ml.Tensor {
	// SAM output shape in ggml: [H, W, C, batch]
	h := samFeatures.Dim(0)
	w := samFeatures.Dim(1)
	hiddenDim := samFeatures.Dim(2) // 896
	batch := samFeatures.Dim(3)
	nPatches := h * w

	// Reshape to [H*W, C, batch] then permute to [C, H*W, batch] = [hiddenDim, seqLen, batch]
	features := samFeatures.Reshape(ctx, nPatches, hiddenDim, batch)
	features = features.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	// Now features: [hiddenDim, nPatches, batch]

	// Select query embeddings based on patch count.
	// We must copy model weight tensors into the current context to avoid
	// cross-context Metal residency set conflicts. Model weight tensors
	// live in the model loading context; using them directly in the
	// EncodeMultimodal compute graph causes Metal assertion failures.
	var srcQuery ml.Tensor
	if nPatches == 144 {
		srcQuery = m.Query768
	} else {
		srcQuery = m.Query1024
	}
	nQueries := srcQuery.Dim(1)

	// Copy query weights into current context as input tensor
	queryData := srcQuery.Floats()
	var queryEmbed ml.Tensor
	if len(queryData) > 0 {
		queryEmbed = ctx.Input().FromFloats(queryData, srcQuery.Dim(0), nQueries)
	} else {
		// Floats() returned nil - cast in the current context to dequantize
		queryEmbed = srcQuery.Cast(ctx, ml.DTypeF32)
	}

	// Expand query for batch: [hiddenDim, nQueries] -> [hiddenDim, nQueries, batch]
	if batch > 1 {
		queryEmbed = queryEmbed.Repeat(ctx, 2, batch)
	}

	queryEmbed = queryEmbed.Cast(ctx, ml.DTypeF32)
	features = features.Cast(ctx, ml.DTypeF32)

	// Concatenate along seqLen dim (Dim 1): [hiddenDim, nPatches+nQueries, batch]
	combined := features.Concat(ctx, queryEmbed, 1)
	seqLen := nPatches + nQueries

	// Build hybrid attention mask
	mask := m.buildHybridMask(ctx, nPatches, nQueries, seqLen)

	// Build positions for RoPE
	positions := ctx.Arange(0, float32(seqLen), 1, ml.DTypeI32)

	// Run through Qwen2 transformer blocks
	// Input/output: [hiddenDim, seqLen, batch]
	hiddenStates := combined
	for _, block := range m.Blocks {
		hiddenStates = block.Forward(ctx, hiddenStates, positions, mask, m.Options)
	}

	// Apply output norm (Qwen2 model's final layer norm)
	if m.OutputNorm != nil {
		hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.Options.eps)
	}

	// Extract only query outputs (causal flow): [hiddenDim, nQueries, batch]
	hiddenStates = hiddenStates.Slice(ctx, 1, nPatches, seqLen, 1)

	return hiddenStates
}

// buildHybridMask creates the hybrid causal/non-causal attention mask.
// Image tokens use bidirectional attention, query tokens use causal attention
// but can attend to all image tokens.
func (m *qwen2Model) buildHybridMask(ctx ml.Context, nImage, nQuery, seqLen int) ml.Tensor {
	// Build mask using same row-by-row pattern as qwen25vl's blockDiagonalMask.
	// s[row][col] → after flatten and FromFloats → ggml mask[col, row]
	// After MulmatFullPrec: kq[key_pos, query_pos]
	// kq.Add(mask) matches when mask ggml[col=key, row=query]
	// So s[row=query][col=key] = 0 means "query can attend to key"
	minVal := float32(math.Inf(-1))

	s := make([][]float32, seqLen)
	for i := range s {
		s[i] = slices.Repeat([]float32{minVal}, seqLen)
	}

	// Image tokens (row=query 0..nImage) can attend to all image keys (bidirectional)
	for q := 0; q < nImage; q++ {
		for k := 0; k < nImage; k++ {
			s[q][k] = 0
		}
	}

	// Query tokens (row=query nImage..seqLen) can attend to:
	// 1. All image keys (cross-attention)
	// 2. Causally to previous/current query keys
	for i := 0; i < nQuery; i++ {
		q := nImage + i
		for k := 0; k < nImage; k++ {
			s[q][k] = 0
		}
		for j := 0; j <= i; j++ {
			k := nImage + j
			s[q][k] = 0
		}
	}

	return ctx.Input().FromFloats(slices.Concat(s...), seqLen, seqLen)
}

type qwen2Block struct {
	AttentionNorm *nn.RMSNorm     `gguf:"attn_norm"`
	Attention     *qwen2Attention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	FeedForward   *qwen2MLP
}

func (m *qwen2Block) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts qwen2Options) ml.Tensor {
	// hiddenStates: [hiddenDim, seqLen, batch]
	residual := hiddenStates
	hiddenStates = m.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.Attention.Forward(ctx, hiddenStates, positions, mask, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = m.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.FeedForward.Forward(ctx, hiddenStates)
	return hiddenStates.Add(ctx, residual)
}

type qwen2Attention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (m *qwen2Attention) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts qwen2Options) ml.Tensor {
	// hiddenStates: [hiddenDim, seqLen, batch]
	// After Linear: [outDim, seqLen, batch]
	// Reshape to: [headDim, numHeads, seqLen, batch]

	query := m.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1), query.Dim(2))

	key := m.Key.Forward(ctx, hiddenStates)
	key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, key.Dim(1), key.Dim(2))

	value := m.Value.Forward(ctx, hiddenStates)
	value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, value.Dim(1), value.Dim(2))

	// Apply RoPE
	query = nn.RoPE(ctx, query, positions, opts.headDim(), opts.ropeTheta, 1.0, rope.WithTypeNeoX())
	key = nn.RoPE(ctx, key, positions, opts.headDim(), opts.ropeTheta, 1.0, rope.WithTypeNeoX())

	scale := 1.0 / math.Sqrt(float64(opts.headDim()))

	// Manual scaled dot-product attention with hybrid mask support
	// nn.Attention doesn't support custom masks when cache=nil,
	// so we implement attention manually (same pattern as qwen25vl).
	// Permute for matmul: [headDim, numHeads, seqLen, batch] -> [headDim, seqLen, numHeads, batch]
	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	// kq = key^T @ query -> [seqLen_k, seqLen_q, numHeads, batch]
	// GQA broadcasting: ggml automatically repeats KV heads to match query heads
	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)

	// Apply hybrid attention mask (image=bidirectional, query=causal)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)

	// kqv = value @ softmax(kq) -> [headDim, seqLen_q, numHeads, batch]
	kqv := value.Mulmat(ctx, kq)

	// Permute back: [headDim, seqLen, numHeads, batch] -> [headDim, numHeads, seqLen, batch]
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	// Reshape: [headDim, numHeads, seqLen, batch] -> [hiddenDim, seqLen, batch]
	attention = attention.Reshape(ctx, -1, attention.Dim(2), attention.Dim(3))
	return m.Output.Forward(ctx, attention)
}

type qwen2MLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (m *qwen2MLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hiddenStates = m.Gate.Forward(ctx, hiddenStates).SILU(ctx, m.Up.Forward(ctx, hiddenStates))
	return m.Down.Forward(ctx, hiddenStates)
}
