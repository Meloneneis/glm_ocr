"""
Single image inference test for GLM-OCR using Hugging Face Transformers.
See: https://huggingface.co/docs/transformers/model_doc/glm_ocr
"""
from unsloth import FastVisionModel
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
import torch
from PIL import Image

base_model_id = "zai-org/GLM-OCR"
adapter_id = "meloneneneis/glm_ocr_21jhd"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(base_model_id, use_fast=False)
model = GlmOcrForConditionalGeneration.from_pretrained(base_model_id)
device = model.device
model.load_adapter(adapter_id)
model.to(device)
FastVisionModel.for_inference(model)


local_path = "/home/meloneneis/PythonProjects/glm_ocr/finetuning/output/BGH_Zivilsenat-13_NA_2020-06-24_XIII_ZB_6_19_NA_NA_0_page_0010.png"
image = Image.open(local_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": image,  # Use "image" key and pass the PIL object
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
output = model.generate(**inputs, max_new_tokens=4096)
text = processor.decode(output[0], skip_special_tokens=True)
# Drop input prompt and image placeholders for cleaner display
if "Text Recognition:" in text:
    text = text.split("Text Recognition:")[-1]
text = text.replace("<think>", "").replace("</think>", "")
text = text.replace("<|image|>", "").strip()

print("Output (OCR text):")
if text:
    for line_no, line in enumerate(text.splitlines(), start=1):
        print(f"  {line_no:4d}  {line}")
else:
    print("(no text)")