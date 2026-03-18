package deepseekocr2

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Model struct {
	model.Base
	tokenizer.Tokenizer

	Sam   *samModel   `gguf:"s"`
	Qwen2 *qwen2Model `gguf:"q"`
	Text  *textModel

	//nolint:misspell // this misspelling is upstream. fixing it breaks the model
	ViewSeperator ml.Tensor `gguf:"mm.view_seperator"`

	Projector *nn.Linear `gguf:"mm.layers"`
}

func (m *Model) EncodeMultimodal(ctx ml.Context, bts []byte) ([]input.Multimodal, error) {
	patches, original, cropInfo, err := ProcessImage(ctx, bts)
	if err != nil {
		return nil, err
	}
	realBlocks := cropInfo[2] // actual number of image patches (rest are zero-padding)

	// Process patches through SAM -> Qwen2 -> Projector
	samOutputs := m.Sam.Forward(ctx, patches)
	qwen2Outputs := m.Qwen2.Forward(ctx, samOutputs)
	localOutputs := m.Projector.Forward(ctx, qwen2Outputs)
	// localOutputs: [1280, nQueries, totalBlocks(padded)]
	// Only use real patches, discard zero-padded ones
	nQueriesLocal := localOutputs.Dim(1)
	localOutputs = localOutputs.Reshape(ctx, localOutputs.Dim(0), -1)
	// Slice to keep only real patch tokens: [1280, nQueries*realBlocks]
	realTokens := nQueriesLocal * realBlocks
	if realTokens < localOutputs.Dim(1) {
		localOutputs = localOutputs.Slice(ctx, 1, 0, realTokens, 1)
	}

	// Process original (global view)
	samOutputs = m.Sam.Forward(ctx, original)
	qwen2Outputs = m.Qwen2.Forward(ctx, samOutputs)
	globalOutputs := m.Projector.Forward(ctx, qwen2Outputs)
	globalOutputs = globalOutputs.Reshape(ctx, globalOutputs.Dim(0), -1)

	// ViewSeperator: [1280] -> needs to be [1280, 1] for concat
	viewSep := m.ViewSeperator.Cast(ctx, ml.DTypeF32)
	viewSep = viewSep.Reshape(ctx, viewSep.Dim(0), 1)

	// Concatenate all vision tokens: [1280, total_tokens]
	// total = nQueries*n_patches + nQueries_base + 1(separator)
	allOutputs := localOutputs.Concat(ctx, globalOutputs, 1)
	allOutputs = allOutputs.Concat(ctx, viewSep, 1)

	return []input.Multimodal{
		{Tensor: allOutputs},
	}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	outputs := make([]*input.Input, 0, len(inputs))
	for i := range inputs {
		if inputs[i].Multimodal == nil {
			outputs = append(outputs, inputs[i])
			continue
		}

		t := inputs[i].Multimodal[0].Tensor
		outputs = append(outputs, &input.Input{
			Token:          128815,
			Multimodal:     inputs[i].Multimodal,
			MultimodalHash: inputs[i].MultimodalHash,
			SameBatch:      t.Dim(1) - 1,
		})

		outputs = slices.Grow(outputs, t.Dim(1)-1)
		outputs = append(outputs, slices.Repeat([]*input.Input{{Token: 128815}}, t.Dim(1)-1)...)
	}
	return outputs, nil
}

func (m *Model) Validate() error {
	// Check if critical Qwen2 tensors are loaded
	if m.Qwen2 == nil {
		return fmt.Errorf("Qwen2 model is nil")
	}
	if m.Qwen2.Query768 == nil {
		return fmt.Errorf("Qwen2 Query768 embedding not loaded")
	}
	if m.Qwen2.Query1024 == nil {
		return fmt.Errorf("Qwen2 Query1024 embedding not loaded")
	}
	if len(m.Qwen2.Blocks) == 0 {
		return fmt.Errorf("Qwen2 has 0 blocks")
	}
	if m.Qwen2.Blocks[0].Attention == nil || m.Qwen2.Blocks[0].Attention.Query == nil {
		return fmt.Errorf("Qwen2 block 0 attention weights not loaded")
	}
	if m.Sam == nil || len(m.Sam.Blocks) == 0 {
		return fmt.Errorf("SAM model not loaded")
	}
	if m.Projector == nil {
		return fmt.Errorf("Projector not loaded")
	}
	return nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	inputsEmbeds := m.Text.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	for _, mm := range batch.Multimodal {
		t := mm.Multimodal[0].Tensor
		ctx.Forward(t.Copy(ctx, inputsEmbeds.View(ctx, mm.Index*inputsEmbeds.Stride(1), t.Dim(0)*t.Dim(1))))
	}

	hiddenStates := inputsEmbeds
	for i, block := range m.Text.Blocks {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.Text.Blocks)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = block.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Text.Options)
	}

	hiddenStates = m.Text.OutputNorm.Forward(ctx, hiddenStates, m.Text.Options.eps)
	return m.Text.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseekocr2", func(c fs.Config) (model.Model, error) {
		textBlocks := make([]textBlock, c.Uint("block_count"))
		leadingDenseBlockCount := int(c.Uint("leading_dense_block_count", 1))
		for i := range textBlocks {
			if i >= leadingDenseBlockCount {
				textBlocks[i].FeedForward = &textMoe{}
			} else {
				textBlocks[i].FeedForward = &textMLP{}
			}
		}

		m := Model{
			Tokenizer: tokenizer.NewBytePairEncoding(
				&tokenizer.Vocabulary{
					Values: c.Strings("tokenizer.ggml.tokens"),
					Types:  c.Ints("tokenizer.ggml.token_type"),
					Merges: c.Strings("tokenizer.ggml.merges"),
					AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
					BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
					AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
					EOS: append(
						[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
						c.Ints("tokenizer.ggml.eos_token_ids")...,
					),
				},
				"\\p{N}{1,3}",
				`[一-龥぀-ゟ゠-ヿ]+`,
				"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
			),
			Text: &textModel{
				Blocks: textBlocks,
				Options: textOptions{
					hiddenSize:     int(c.Uint("embedding_length")),
					numHeads:       int(c.Uint("attention.head_count")),
					numKVHeads:     int(c.Uint("attention.head_count_kv")),
					numExperts:     int(c.Uint("expert_count")),
					numExpertsUsed: int(c.Uint("expert_used_count")),
					ropeBase:       c.Float("rope.freq_base", 10_000),
					ropeScale:      c.Float("rope.scaling.factor", 1.0),
					eps:            c.Float("attention.layer_norm_rms_epsilon", 1e-6),
				},
			},
			Qwen2: &qwen2Model{
				Blocks: make([]qwen2Block, c.Uint("qwen2.block_count", 24)),
				Options: qwen2Options{
					hiddenSize:       int(c.Uint("qwen2.embedding_length", 896)),
					numHeads:         int(c.Uint("qwen2.head_count", 14)),
					numKVHeads:       int(c.Uint("qwen2.head_count_kv", 2)),
					intermediateSize: int(c.Uint("qwen2.intermediate_size", 4864)),
					eps:              c.Float("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
					ropeTheta:        c.Float("qwen2.rope.freq_base", 1_000_000),
				},
			},
			Sam: &samModel{
				Blocks: make([]samBlock, c.Uint("sam.block_count")),
				Options: samOptions{
					hiddenSize:            int(c.Uint("sam.embedding_length")),
					numHeads:              int(c.Uint("sam.head_count")),
					eps:                   c.Float("sam.attention.layer_norm_epsilon", 1e-6),
					globalAttentionLayers: c.Ints("sam.global_attention_indexes"),
				},
			},
		}

		m.Cache = kvcache.NewCausalCache(m.Text.Shift)
		return &m, nil
	})
}
