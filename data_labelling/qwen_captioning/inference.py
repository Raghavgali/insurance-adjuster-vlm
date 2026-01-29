from accelerate import Accelerator
from src.model_loader import load_qwen_model
from src.processor import get_stopping_criteria, build_message
from src.utils import load_images_from_folder, save_jsonl, log, load_image
from qwen_vl_utils import process_vision_info
import os 
import torch
import json

model, processor, tokenizer = load_qwen_model()
accelerator = Accelerator()
model = accelerator.prepare(model)

batch_size = 1

def process_batch(messages_list, vision_inputs, video_inputs, image_paths, output_folder):
    prompts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]

    inputs = processor(
        text=prompts,
        images=vision_inputs,
        return_tensors="pt",
        padding=True
    ).to(accelerator.device)

    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    stopping_criteria = get_stopping_criteria(tokenizer)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            stopping_criteria=stopping_criteria,
            max_new_tokens=512
        )

    for img_path, input_ids, gen_ids in zip(image_paths, inputs["input_ids"], generated_ids):
        trimmed = gen_ids[len(input_ids):]
        caption = processor.decode(trimmed, skip_special_tokens=True).strip()

        # Fix 1: Add fallback
        if not caption:
            caption = "[No damage detected or model failed to generate output]"

        # Fix 2: Log empty caption
        if caption == "[No damage detected or model failed to generate output]":
            log(f"[WARNING] Empty caption for {img_path.name}, using fallback.")

        # Fix 3: Optional skip (comment out if fallback is always desired)
        if caption == "[No damage detected or model failed to generate output]":
            log(f"[SKIPPED] Empty caption for {img_path.name}")
            continue

        result = {"image": str(img_path.name), "report": caption}
        output_file = os.path.join(output_folder, f"{img_path.stem}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        log(f"Saved caption for {img_path.name}")

def run_captioning_on_folder(input_folder, output_folder, processor, tokenizer):
    all_images = load_images_from_folder(input_folder)
    done = {
        f.stem for f in os.listdir(output_folder)
        if f.suffix == '.json'
    }
    image_paths = [p for p in all_images if p.stem not in done]
    os.makedirs(output_folder, exist_ok=True)

    batch_messages = []
    batch_vision_inputs = []
    batch_video_inputs = []
    batch_image_paths = []

    for img_path in image_paths:
        log(f"Preparing image: {img_path.name}")
        image = load_image(img_path)
        messages = build_message(image)
        vision_input, video_input = process_vision_info(messages)

        batch_messages.append(messages)
        batch_vision_inputs.append(vision_input)
        batch_video_inputs.append(video_input)
        batch_image_paths.append(img_path)

        if len(batch_messages) == batch_size:
            process_batch(batch_messages, batch_vision_inputs, batch_video_inputs, batch_image_paths, output_folder)
            batch_messages, batch_vision_inputs, batch_video_inputs, batch_image_paths = [], [], [], []

    if batch_messages:
        process_batch(batch_messages, batch_vision_inputs, batch_video_inputs, batch_image_paths, output_folder)

# Run captioning on all splits 
for split in ["train", "valid", "test"]:
    log(f"Processing split: {split}")
    input_folder = os.path.join("data", "raw", split)
    output_folder = os.path.join("data", "captions", split)
    run_captioning_on_folder(input_folder, output_folder, processor, tokenizer)