import os 
import json
import yaml
import requests
# from huggingface_hub import InferenceClient
from openai import OpenAI

with open('/Users/raghavg/Desktop/Datasets and Projects/Insurance/config/openai_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
api_key = config['api_key']

client = OpenAI(api_key=api_key)

def filtering(json_path, output_path):
    results = []
    total_lines = sum(1 for _ in open(json_path, 'r'))
    with open(json_path, 'r', encoding='utf-8') as json_file:
        print(f"[START] Filtering Started on {total_lines} files.")
        eval_count = 0
        for line in json_file:
            data = json.loads(line)
            img_name = data.get("image", "")
            original_caption = data.get("caption", "")
            evaluation = data.get("evaluation", "")

            prompt = f"""
                "You are a filtering agent. Based on the following evaluation, decide if the caption needs to be removed from the dataset. "
                "Respond only with 'yes' or 'no'.\n"
                f"Evaluation:\n{evaluation}\n"
                "Should the caption be removed?"
            """
            response = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [{
                    "role": "user", "content": prompt
                }],
                timeout=20
            )

            result = {
                "image": img_name,
                "original_caption": original_caption,
                "original_evaluation": evaluation,
                "filtering_response": response.choices[0].message.content.strip().lower()
            }
            results.append(result)
            eval_count += 1
            print(f"[INTERMEDIATE] Filtering done on {eval_count}/{total_lines}")

    with open(output_path, 'w', encoding='utf-8') as filtering:
        for r in results:
            filtering.write(json.dumps(r) + "\n")

    print(f"[DONE] Filtering done.")

if __name__ == "__main__":
    input_json = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/captions_evaluations.jsonl"
    output_json = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/post-filtering_evaluations.jsonl"
    filtering(input_json, output_json)