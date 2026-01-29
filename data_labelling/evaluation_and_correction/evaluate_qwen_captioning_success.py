import os 
import json 


def count_filter_response(filepath):
    counts = {'yes': 0, 'no': 0}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try: 
                response = json.loads(line).get("filtering_response", "").strip().lower()
                if response in counts:
                    counts[response] += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping Malformed line:\n{line}")
    return counts


if __name__=='__main__':
    filepath = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/evaluation_and_correction/post-filtering_evaluations.jsonl'
    result = count_filter_response(filepath=filepath)
    print(f"Yes: {result['yes']}, No: {result['no']}")

