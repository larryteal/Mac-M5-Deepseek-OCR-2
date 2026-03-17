"""
Patches the cached DeepSeek-OCR-2 model source code for Mac compatibility.

Run this after the model has been downloaded (e.g., after a first failed run).
It modifies the cached modeling file at:
    ~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR-2/.../modeling_deepseekocr2.py

Changes made:
    1. Replace all .cuda() calls with ._target_device attribute lookup
    2. Replace torch.bfloat16 image preprocessing with torch.float32
    3. Replace torch.autocast("cuda") with torch.no_grad()
    4. Add cross-device tensor migration in DeepseekOCR2Model.forward()
    5. Add MPS-safe embedding computation path
"""

import glob
import os
import shutil
import sys


def find_model_file():
    """Find the cached modeling_deepseekocr2.py file."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    pattern = os.path.join(cache_dir, "**/DeepSeek*OCR*2/**/modeling_deepseekocr2.py")
    matches = glob.glob(pattern, recursive=True)

    # Filter out backup files
    matches = [m for m in matches if not m.endswith(".bak")]

    if not matches:
        print("ERROR: Model file not found. Please run the model once first to download it:")
        print('  python -c "from transformers import AutoModel; AutoModel.from_pretrained(\'deepseek-ai/DeepSeek-OCR-2\', trust_remote_code=True)"')
        sys.exit(1)

    return matches[0]


def patch_file(filepath):
    """Apply all patches to the model file."""
    # Create backup
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"Backup created: {backup}")
    else:
        print(f"Backup already exists: {backup}")

    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Patch 1: Replace .cuda() with device-aware attribute
    content = content.replace(
        ".cuda()",
        ".to(self._target_device if hasattr(self, '_target_device') else 'cuda')"
    )

    # Patch 2: Replace bfloat16 image preprocessing with float32
    content = content.replace(".to(torch.bfloat16)", ".to(torch.float32)")

    # Patch 3: Replace autocast("cuda") with no_grad()
    content = content.replace(
        'with torch.autocast("cuda", dtype=torch.bfloat16):',
        "with torch.no_grad():  # autocast disabled for MPS compatibility"
    )

    # Patch 4: Add cross-device tensor migration before LLM forward
    # Find the super().forward() call and add device migration before it
    old_super_call = """        return super(DeepseekOCR2Model, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,"""

    new_super_call = """        # Move all tensors to LLM device after vision encoding on CPU
        _llm_device = getattr(self, '_llm_device', None)
        if _llm_device:
            if inputs_embeds is not None and inputs_embeds.device.type != _llm_device:
                inputs_embeds = inputs_embeds.to(_llm_device)
            if position_ids is not None and position_ids.device.type != _llm_device:
                position_ids = position_ids.to(_llm_device)
            if attention_mask is not None and attention_mask.device.type != _llm_device:
                attention_mask = attention_mask.to(_llm_device)
            if images_seq_mask is not None and images_seq_mask.device.type != _llm_device:
                images_seq_mask = images_seq_mask.to(_llm_device)

        return super(DeepseekOCR2Model, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,"""

    content = content.replace(old_super_call, new_super_call)

    # Patch 5: Add MPS-safe embedding path
    old_embed = """        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)"""

    new_embed = """        if inputs_embeds is None:
            # For MPS hybrid: do embedding on CPU if vision encoding needed
            _vision_device = getattr(self, '_vision_device', None)
            if _vision_device == 'cpu' and input_ids.device.type != 'cpu':
                inputs_embeds = self.get_input_embeddings().to('cpu')(input_ids.to('cpu'))
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)"""

    content = content.replace(old_embed, new_embed)

    if content == original:
        print("File appears to already be patched (no changes needed)")
        return False

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Patched: {filepath}")
    return True


def main():
    filepath = find_model_file()
    print(f"Found model file: {filepath}")

    if patch_file(filepath):
        print("\nPatching complete! You can now run:")
        print("  python ocr_hybrid.py image.png")
    else:
        print("\nNo changes were needed.")


if __name__ == "__main__":
    main()
