import os 
import json 
import pandas as pd 
import time 
import tiktoken
import yaml

with open("/Users/raghavg/Desktop/Datasets and Projects/Insurance/evaluation_and_correction/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
api_key = config["api_key"]

from openai import OpenAI
client = OpenAI(api_key = api_key)
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def num_tokens(text):
    return len(tokenizer.encode(text))

def evaluation(captions_dir, meta_data_file, output_file, log_file):
    filenames = [f for f in os.listdir(captions_dir) if f.endswith(".json")]
    meta_data_df = pd.read_csv(meta_data_file)
    results = []
    total_tokens = 0
    eval_count = 0
    start_time = time.time()

    with open(log_file, "w") as log:
        total_files = len(filenames)
        log.write(f"[START] Evaluation started on {total_files} files\n")

        for idx, json_fname in enumerate(filenames):
            img_name = os.path.splitext(json_fname)[0] + '.jpg'

            time.sleep(21)

            try:
                with open(os.path.join(captions_dir, json_fname), 'r') as f:
                    img_caption = json.load(f)
            except json.JSONDecodeError:
                log.write(f"[ERROR] Failed to load caption: {json_fname}\n")
                continue

            # Matching metadata 
            meta_row = meta_data_df[meta_data_df['file_name'] == img_name]
            if meta_row.empty:
                log.write(f"[SKIP] No metadata for {img_name}\n")
                continue

            row = meta_row.iloc[0]
            caption_text = img_caption.get("report", "").strip()

            prompt = f"""
You are an expert Insurance Adjuster, you are evaluating a vehicle damage report for accuracy against metadata.

### Report:
{caption_text}

### Metadata:
- Shooting Angle: {row['shooting angle']}
- View: {row['complete or partial ']}
- Color: {row['color']}
- Damage Category: {row['category_name']}
- Area: {row['area']}
- BBox : {row['bbox']}
- Segmentation Length: {len(json.loads(row['segmentation']) if isinstance(row['segmentation'], str) else row['segmentation'])}
- IsCrowd: {row['iscrowd']}

### Task:
Based on the metadata above, evaluate the accuracy of the caption. Answer:
- Is the caption consistent with the damage category and metadata?
- If not, briefly explain what's incorrect.
- Rate the caption out of 5. 

Respond in one paragraph
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
                    "caption": img_caption.get("report", "NOT PROVIDED"),
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
    captions_dir = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/captions/train"
    meta_data_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/enriched_trained_meta.csv"
    output_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/captions_evaluations.jsonl"
    log_file = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/captions_log.txt"

    evaluation(captions_dir=captions_dir,
               meta_data_file=meta_data_file,
               output_file=output_file,
               log_file=log_file)