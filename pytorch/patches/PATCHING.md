# Model Source Code Patches

DeepSeek-OCR-2's HuggingFace model code is designed for CUDA GPUs only. To run on Mac, the following patches are needed on the cached file:

**File**: `~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR-2/.../modeling_deepseekocr2.py`

## Patch 1: Replace `.cuda()` calls

The model hardcodes `.cuda()` in ~9 places. Replace all with a device-aware attribute:

```diff
- .cuda()
+ .to(self._target_device if hasattr(self, '_target_device') else 'cuda')
```

## Patch 2: Replace `torch.bfloat16` image preprocessing

Image tensors are hardcoded to bfloat16, which causes dtype mismatches on non-CUDA devices:

```diff
- .to(torch.bfloat16)
+ .to(torch.float32)
```

## Patch 3: Disable `torch.autocast("cuda")`

MPS does not support CUDA autocast:

```diff
- with torch.autocast("cuda", dtype=torch.bfloat16):
+ with torch.no_grad():  # autocast disabled for MPS compatibility
```

## Patch 4: Cross-device tensor migration

When using hybrid CPU+MPS, tensors computed on CPU must be moved to MPS before the LLM forward pass:

```python
# Add before super().forward() in DeepseekOCR2Model.forward():
_llm_device = getattr(self, '_llm_device', None)
if _llm_device:
    if inputs_embeds is not None and inputs_embeds.device.type != _llm_device:
        inputs_embeds = inputs_embeds.to(_llm_device)
    if position_ids is not None and position_ids.device.type != _llm_device:
        position_ids = position_ids.to(_llm_device)
    if attention_mask is not None and attention_mask.device.type != _llm_device:
        attention_mask = attention_mask.to(_llm_device)
```

## Patch 5: MPS-safe embedding computation

When vision encoder is on CPU but embed_tokens is also on CPU, ensure input_ids are on the correct device:

```python
# Replace the inputs_embeds computation:
_vision_device = getattr(self, '_vision_device', None)
if _vision_device == 'cpu' and input_ids.device.type != 'cpu':
    inputs_embeds = self.get_input_embeddings().to('cpu')(input_ids.to('cpu'))
else:
    inputs_embeds = self.get_input_embeddings()(input_ids)
```

## Automatic Patching

Run `python apply_patch.py` to apply all patches automatically.
A backup of the original file is created at `modeling_deepseekocr2.py.bak`.
