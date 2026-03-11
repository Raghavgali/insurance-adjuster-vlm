# Data Labelling Pipeline

This directory contains the complete pipeline used to generate, evaluate, and filter training captions for the insurance damage assessment dataset. Two parallel captioning systems (GPT-4o and Qwen2.5-VL) were used, with GPT-4o-mini serving as the quality evaluator for both.

## Pipeline Overview

```
Raw Images (887) + Metadata (CSV)
        │
        ├──→ GPT-4o Captioning
        │       │
        │       ▼
        │    GPT-4o-mini Evaluation (rate 1-5, recommend keep/remove)
        │       │
        │       ▼
        │    Post-filtering → 734 accepted / 153 rejected
        │
        └──→ Qwen2.5-VL-7B Captioning
                │
                ▼
             GPT-4o-mini Evaluation (same criteria)
                │
                ▼
             Post-filtering
```

The GPT pipeline produced the final training dataset (734 samples after filtering).

## Directory Structure

```
data_labelling/
├── config/                              # API keys and model configs (gitignored)
│   ├── openai_config.yaml
│   ├── huggingface_config.yaml
│   └── config.yaml
├── gpt_captioning/
│   ├── captioning.py                    # GPT-4o caption generation
│   ├── utils.py                         # Image encoding, metadata loading
│   └── final_dataset_preparation.py     # Convert filtered captions → training format
├── qwen_captioning/
│   ├── inference.py                     # Qwen2.5-VL inference loop
│   ├── model_loader.py                  # 8-bit quantized model loading
│   ├── processor.py                     # Chat template + stopping criteria
│   └── utils.py                         # Helpers
├── evaluation_and_correction/
│   ├── evaluation.ipynb                 # Metadata enrichment notebook
│   ├── stage_one_evaluation_gpt.py      # Evaluate GPT captions with GPT-4o-mini
│   ├── stage_one_evaluation_qwen.py     # Evaluate Qwen captions with GPT-4o-mini
│   ├── post_evaluation_filtering_gpt.py # Parse evaluations → keep/remove
│   ├── post-evaluation_filtering_qwen.py
│   └── evaluate_qwen_captioning_success.py
├── data/
│   ├── metadata/enriched_trained_meta.csv
│   ├── gpt/                             # GPT pipeline outputs
│   │   ├── captioning.jsonl
│   │   ├── gpt_captions_evaluations.jsonl
│   │   ├── filtered_captions.jsonl
│   │   └── removed_captions.jsonl
│   └── qwen/captions/train/            # One JSON per image
├── logs/                                # Execution logs with token counts
└── misc/                                # One-off utilities
```

## Stage 1: Metadata Enrichment

Before captioning, raw COCO-format annotations were enriched into a flat CSV (`enriched_trained_meta.csv`) using the `evaluation.ipynb` notebook. Each row contains:

| Field | Example | Source |
|-------|---------|--------|
| file_name | 001516.jpg | COCO annotations |
| shooting angle | side / front / rear / Unknown | Annotation attribute |
| complete or partial | partial | Annotation attribute |
| color | red | Annotation attribute |
| category_name | dent | Mapped from category ID (1=dent, 2=scratch, 3=crack, 4=glass shatter, 5=lamp broken, 6=tire flat) |
| area | 4524.0 | COCO segmentation area |
| bbox | [349.68, 269.34, 339.79, 19.44] | COCO bounding box |
| iscrowd | 0 | COCO crowd flag |

## Stage 2: Caption Generation

### GPT-4o Pipeline

**Script**: `gpt_captioning/captioning.py`

Each image was base64-encoded and sent to GPT-4o along with its metadata row. The model was instructed to write a one-paragraph damage report including location, type, and estimated repair cost.

| Setting | Value |
|---------|-------|
| Model | gpt-4o |
| Temperature | 0.5 |
| Timeout | 20s per call |
| Images processed | 887 |

**Prompt**:
```
You are an expert Insurance Adjuster. Given this image and metadata,
write a short but detailed one-paragraph damage report with location, type,
and estimated insurance-covered cost, or say 'No Damage'.

Metadata:
- Shooting Angle: {shooting_angle}
- View: {complete_or_partial}
- Color: {color}
- Damage Category: {category_name}
- Area: {area}
- BBox: {bbox}
- Segmentation Length: {seg_length}
- IsCrowd: {iscrowd}
```

**Output**: `data/gpt/captioning.jsonl` (887 records)

### Qwen2.5-VL Pipeline

**Script**: `qwen_captioning/inference.py`

The open-source Qwen2.5-VL-7B-Instruct model was used as a secondary captioner. It received only the image (no metadata) and a simpler prompt.

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-VL-7B-Instruct |
| Quantization | 8-bit (BitsAndBytes) |
| Max new tokens | 512 |
| Stopping criteria | EOS token |

**Prompt**:
```
You are an Insurance adjuster. Complete a damage report for this vehicle
in this image. Include details about the damage, including the location
and type of damage. If there is no damage, say 'No damage'.
```

**Output**: One JSON file per image in `data/qwen/captions/train/`

## Stage 3: Quality Evaluation

Both caption sets were evaluated using GPT-4o-mini, which compared each caption against the ground-truth metadata.

**Scripts**: `stage_one_evaluation_gpt.py`, `stage_one_evaluation_qwen.py`

| Setting | Value |
|---------|-------|
| Model | gpt-4o-mini |
| Temperature | 0.2 |
| Rate limiting | 21s between calls |

**Evaluation prompt** (for each caption):
```
You are an expert Insurance Adjuster evaluating a vehicle damage report
for accuracy against metadata.

### Evaluation Instructions:
1. Rate the caption from 1 to 5.
2. Comment on the consistency of the caption with metadata.
3. Should this caption be removed? Respond with 'yes' or 'no'.
```

**GPT evaluation cost**: ~$0.16 total (328,737 tokens over ~5.8 hours)

## Stage 4: Post-Filtering

**Scripts**: `post_evaluation_filtering_gpt.py`, `post-evaluation_filtering_qwen.py`

Evaluation responses were parsed to extract the keep/remove decision. The script searches for the "should this caption be removed?" pattern and extracts the yes/no answer, falling back to the last yes/no found in the text.

### GPT Pipeline Results

| Metric | Count |
|--------|-------|
| Total captions evaluated | 887 |
| Accepted (keep) | 734 (82.8%) |
| Rejected (remove) | 153 (17.2%) |

## Stage 5: Final Dataset Preparation

**Script**: `gpt_captioning/final_dataset_preparation.py`

The 734 accepted GPT captions were converted into the conversation format used for model training:

```json
{
  "id": "001516",
  "image": "001516.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "You are an Insurance Adjuster. Evaluate the car damage...\n\nMetadata:\n- Shooting Angle: front\n..."
    },
    {
      "role": "assistant",
      "content": "The image depicts a red vehicle with a noticeable dent..."
    }
  ]
}
```

This produced `dataset/dataset.json` (734 samples), which was later cleaned (metadata removed from prompts) by `GLM/data/dataset_cleanup.py` for image-only training.

## Why Two Captioners?

GPT-4o was the primary captioner due to higher caption quality. Qwen2.5-VL was explored as a cost-free open-source alternative. Both went through the same evaluation pipeline, but the final training dataset uses GPT-4o captions because they scored higher on the metadata-consistency evaluations.
