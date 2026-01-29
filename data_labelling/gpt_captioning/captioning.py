import os
import json 
from openai import OpenAI
import yaml 
import pandas as pd 
import tiktoken
import base64
from utils import load_images_from_folder, load_image, save_jsonl

with open('/Users/raghavg/Desktop/Datasets and Projects/Insurance/config/openai_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
api_key = config['api_key']

client = OpenAI(api_key=api_key)
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def num_tokens(text):
    return len(tokenizer.encode(text))

def captioning(images_path, output_path, meta_data_file, log):
    results = []
    meta_data_df = pd.read_csv(meta_data_file)
    image_files = load_images_from_folder(images_path)

    with open(log, 'a') as log_file:
        log_file.write(f"[START] Captioning started on {len(image_files)}\n")
    eval_count = 1 

    for img_path in image_files:
        img_name = os.path.basename(img_path)

        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        with open(log, 'a') as log_file:
            log_file.write(f"[INTERMEDIATE] Captioning done on {eval_count}/{len(image_files)}\n")

        meta_row = meta_data_df[meta_data_df['file_name'] == img_name]
        if meta_row.empty:
            with open(log, 'a') as log_file:
                log_file.write(f"[SKIP] No metadata for image: {img_name}\n")
            continue

        row = meta_row.iloc[0]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert Insurance Adjuster. Given this image and metadata, "
                            "write a short but detailed one‑paragraph damage report with location, type, "
                            "and estimated insurance‑covered cost, or say 'No Damage'.\n\n"
                            f"Metadata:\n"
                            f"- Shooting Angle: {row['shooting angle']}\n"
                            f"- View: {row['complete or partial ']}\n"
                            f"- Color: {row['color']}\n"
                            f"- Damage Category: {row['category_name']}\n"
                            f"- Area: {row['area']}\n"
                            f"- BBox: {row['bbox']}\n"
                            f"- Segmentation Length: {len(json.loads(row['segmentation']) if isinstance(row['segmentation'], str) else row['segmentation'])}\n"
                            f"- IsCrowd: {row['iscrowd']}"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        with open(log, 'a') as log_file:
            log_file.write(f"[DEBUG] Sending image: {img_name}, metadata length: {len(messages[0]['content'][0]['text'])}, base64 size: {len(base64_image)}\n")

        input_tokens = num_tokens(json.dumps(messages))
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                timeout=20
            )

            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens_used = usage.total_tokens

            report = response.choices[0].message.content.strip()

            result = {
                "image": img_name,
                "report": report
            }

            results.append(result)
            with open(log, 'a') as log_file:
                log_file.write(f"[SUCCESS] {img_name} | input: {input_tokens}, output: {output_tokens}, total_tokens: {total_tokens_used}\n")

            eval_count += 1
        except Exception as e:
            with open(log, 'a') as log_file:
                log_file.write(f"[ERROR] {img_name} | {str(e)}\n")
            continue

    save_jsonl(results, output_path)
    with open(log, 'a') as log_file:
        log_file.write(f"[DONE] Captioning done on {len(results)} images.\n")


if __name__ == '__main__':
    images_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/images/train"
    output_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/captioning.jsonl"
    meta_data_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/metadata/enriched_trained_meta.csv"
    log = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/logs/gpt_logs.txt"
    captioning(images_path=images_path,
               output_path=output_path,
               meta_data_file=meta_data_file,
               log=log)