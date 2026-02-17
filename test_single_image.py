"""
Single image inference test for GLM-OCR using Hugging Face Transformers.
See: https://huggingface.co/docs/transformers/model_doc/glm_ocr
"""
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
import torch

model_id = "zai-org/GLM-OCR"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
model = GlmOcrForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
            },
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]

print("Preparing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

print("Generating...")
output = model.generate(**inputs, max_new_tokens=512)
text = processor.decode(output[0], skip_special_tokens=True)
# Drop input prompt and image placeholders for cleaner display
if "Text Recognition:" in text:
    text = text.split("Text Recognition:")[-1]
text = text.replace("<|image|>", "").strip()
print("Output (OCR text):")
print(text or "(no text)")
