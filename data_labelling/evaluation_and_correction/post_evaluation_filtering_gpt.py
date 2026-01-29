import json 
import re
import os 
import sys

def extract_answer(evaluation: str) -> str:
    evaluation = evaluation.lower()
    lines = evaluation.splitlines()

    for line in reversed(lines):
        line = line.replace("**", "").strip()
        if "should this caption be removed?" in line:
            match = re.search(r"\b(yes|no)\b", line)
            if match:
                return match.group(1)
            
    fallback = re.findall(r"\b(yes|no)\b", evaluation)
    if fallback:
        return fallback[-1]
    return None


def filtering(captions_path, output_path, removed_path):
    counts = {'yes': 0, 'no': 0}

    with open(captions_path, 'r', encoding='utf-8') as r, open(output_path, 'w', encoding='utf-8') as w, \
        open(removed_path, 'w', encoding='utf-8') as m:

        for line in r: 
            data = json.loads(line)
            evaluation = data.get('evaluation', '')
            answer = extract_answer(evaluation)
            if answer in counts:
                counts[answer] += 1
                if answer == 'no':
                    w.write(json.dumps(data, ensure_ascii=False) + "\n")
                else: 
                    m.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                print(f"[WARN] Unknown answer '{answer}' in record: {line.strip()}", file=sys.stderr)
        else:
            print(f"[WARN] No match in evaluation: {evaluation!r}", file=sys.stderr)

    print(f"Summary: Yes: {counts['yes']}, No: {counts['no']}")
    return counts


if __name__ == "__main__":
    captions_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/gpt_captions_evaluations.jsonl"
    output_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/filtered_captions.jsonl"
    removed_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/removed_captions.jsonl"
    filtering(captions_path=captions_path,
              output_path=output_path,
              removed_path=removed_path)