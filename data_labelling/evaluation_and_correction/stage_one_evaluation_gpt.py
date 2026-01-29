import os 
import json 
import pandas as pd 
import time 
import tiktoken
import yaml

with open("/Users/raghavg/Desktop/Datasets and Projects/Insurance/config/openai_config.yaml", "r") as file:
    config = yaml.safe_load(file)
api_key = config["api_key"]

from openai import OpenAI
client = OpenAI(api_key = api_key)
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def num_tokens(text):
    return len(tokenizer.encode(text))

def evaluation(captions_file, meta_data_file, output_file, log_file):
    meta_data_df = pd.read_csv(meta_data_file)
    results = []
    total_tokens = 0
    eval_count = 0
    start_time = time.time()

    with open(log_file, "w") as log:
        with open(captions_file, 'r') as f:
            lines = f.readlines()
        total_files = len(lines)
        log.write(f"[START] Evaluation started on {total_files} files\n")

        for line in lines:
            data = json.loads(line)
            img_name = data.get("image", "")
            report = data.get("report", "")

            time.sleep(21)

            # Matching metadata 
            meta_row = meta_data_df[meta_data_df['file_name'] == img_name]
            if meta_row.empty:
                log.write(f"[SKIP] No metadata for {img_name}\n")
                continue

            row = meta_row.iloc[0]
            segmentation = row['segmentation']
            if isinstance(segmentation, str):
                segmentation = json.loads(segmentation)
            prompt = f"""
You are an expert Insurance Adjuster evaluating a vehicle damage report for accuracy against metadata.

### Report:
{report}

### Metadata:
- Shooting Angle: {row['shooting angle']}
- View: {row['complete or partial ']}
- Color: {row['color']}
- Damage Category: {row['category_name']}
- Area: {row['area']}
- BBox: {row['bbox']}
- Segmentation Length: {len(segmentation)}
- IsCrowd: {row['iscrowd']}


### Evaluation Instructions:
1. Rate the caption from 1 to 5.
2. Comment on the consistency of the caption with metadata.
3. Should this caption be removed? Respond with 'yes' or 'no'.

"""
            input_tokens = num_tokens(prompt)

            try:
                response = client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = [{"role": "user", "content": prompt}],
                    temperature=0.2,
                    timeout=20
                )
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens_used = usage.total_tokens

                result = {
                    "image": img_name,
                    "caption": report,
                    "evaluation": response.choices[0].message.content
                }
                results.append(result)
                eval_count += 1
                total_tokens += total_tokens_used
                log.write(f"[EVAL] {img_name} | Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens_used}\n")
                log.write(f"[PROGRESS] Completed {eval_count}/{total_files} files\n")
                log.flush()
            except Exception as e:
                log.write(f"[ERROR] API call failed for {img_name}: {str(e)}\n")

    # Saving output
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    cost = (total_tokens / 1000) * 0.0005
    duration = time.time() - start_time

    with open(log_file, "a") as log:
        log.write(f"\nTotal Tokens: {total_tokens}\n")
        log.write(f"Estimated Cost: ${cost:.4f}\n")
        log.write(f"Total Time: {duration:.2f} sec\n")

    print(f"[DONE] Evaluated {eval_count} captions. Estimated cost: ${cost:.4f}")

if __name__ == '__main__':
    captions_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/captioning.jsonl"
    meta_data_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/metadata/enriched_trained_meta.csv"
    output_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/gpt_captions_evaluations.jsonl"
    log_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/logs/gpt_caption_evaluation_log.txt"

    evaluation(captions_file=captions_file,
               meta_data_file=meta_data_file,
               output_file=output_file,
               log_file=log_file)