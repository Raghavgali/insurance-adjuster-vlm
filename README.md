# Insurance Damage Assessment with GLM-4.6V-Flash

An end-to-end vision-language fine-tuning pipeline for generating insurance-style vehicle damage assessments from a single image. The current stack uses `zai-org/GLM-4.6V-Flash`, LoRA, low-bit loading, DDP training, structured evaluation, and Runpod-oriented deployment scripts.

## What This Project Does

Given one vehicle image, the model is fine-tuned to generate a professional assessment covering:

- visible damage type
- approximate damage location
- qualitative severity
- estimated repair cost range

The deployment target is image-only inference. Metadata is not injected into prompts during training. Annotation-derived metadata is retained for evaluation and error analysis.

## Current Model Stack

- Base model: `zai-org/GLM-4.6V-Flash`
- Fine-tuning method: LoRA via PEFT
- Quantization: 8-bit BitsAndBytes
- Training runtime: PyTorch + DDP
- Tracking: Weights & Biases
- Dataset storage: Hugging Face dataset repo
- Target infra: Runpod on 4x L40 GPUs

## Project Layout

```text
Insurance/
├── GLM/
│   ├── configs/
│   │   ├── download_dataset.yaml
│   │   └── runpod.yaml
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataset_cleanup.py
│   │   ├── download_dataset.py
│   │   ├── sampler.py
│   │   ├── train_test_split.py
│   │   ├── upload_dataset.py
│   │   └── validate_data.py
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── io.py
│   │   ├── prediction_schema.py
│   │   └── regression_metrics.py
│   ├── scripts/
│   │   ├── collator.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── model_loader.py
│   │   ├── train.py
│   │   └── utils/
│   │       ├── hf_utils.py
│   │       ├── load_config.py
│   │       ├── logging.py
│   │       └── wandb.py
│   ├── runpod_ddp.sh
│   └── runpod_setup.sh
├── data_labelling/
│   └── data/
│       └── metadata/
├── dataset/
│   ├── cleaned_dataset.json
│   ├── train.json
│   ├── test.json
│   └── <images>
├── README.md
└── requirements.txt
```

## Data Pipeline

The project uses a cleaned conversational dataset with this structure:

```json
{
  "id": "001516",
  "image": "001516.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "You are an Insurance Adjuster. Evaluate the car damage shown in the image."
    },
    {
      "role": "assistant",
      "content": "The image shows a dent on the front left fender ... estimated repair cost is approximately $800 to $1,200."
    }
  ]
}
```

### Why the dataset is cleaned

The raw prompts and raw captions contained annotation-derived metadata such as:

- damage category
- bbox coordinates
- area
- segmentation-style cues

Those fields create label leakage if the deployment target is image-only inference. The cleaning pass removes that metadata from the user prompt and strips the most direct annotation artifacts from the assistant target while keeping the task semantics intact.

## Dataset Preparation

### 1. Clean the raw dataset

```bash
python3 GLM/data/dataset_cleanup.py dataset/dataset.json dataset/ --dry-run
python3 GLM/data/dataset_cleanup.py dataset/dataset.json dataset/
```

This writes `dataset/cleaned_dataset.json`.

### 2. Create train/test split

```bash
python3 GLM/data/train_test_split.py dataset/cleaned_dataset.json dataset/ --dry-run
python3 GLM/data/train_test_split.py dataset/cleaned_dataset.json dataset/
```

### 3. Validate dataset structure

```bash
python3 GLM/data/validate_data.py --input dataset/train.json
python3 GLM/data/validate_data.py --input dataset/test.json
```

### 4. Upload dataset to Hugging Face

```bash
export HF_TOKEN="<your_token>"

python3 GLM/data/upload_dataset.py \
  --repo-id <your-username>/<your-dataset-repo> \
  --dataset-dir dataset \
  --revision main \
  --private \
  --commit-message "Upload cleaned insurance dataset"
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` with the values you actually use.

## Training Workflow

### 1. Review the training config

Main config: `GLM/configs/runpod.yaml`

Key fields to verify before launch:

- `model.model_id`
- `data.train_annotation_path`
- `data.test_annotation_path`
- `data.image_root`
- `wandb.*`

The config is already set up for `GLM-4.6V-Flash` with:

- chat-template based prompting
- `transformers>=5.0.0rc0`
- `lora.target_modules: "all-linear"`

### 2. Smoke test locally or on one GPU

```bash
python3 GLM/scripts/train.py \
  --config GLM/configs/runpod.yaml \
  --output-dir outputs/smoke_test \
  --epochs 1 \
  --max-steps 5
```

### 3. Run on Runpod

Bootstrap the environment:

```bash
bash GLM/runpod_setup.sh
```

If you want dataset auto-download during setup:

```bash
AUTO_DOWNLOAD_DATASET=1 bash GLM/runpod_setup.sh
```

Launch DDP training:

```bash
bash GLM/runpod_ddp.sh
```

## Evaluation

```bash
python3 GLM/scripts/evaluate.py \
  --config GLM/configs/runpod.yaml \
  --checkpoint outputs/run_name/last.pt \
  --split test \
  --output-dir outputs/eval
```

Implemented evaluation components:

- loss and token accuracy
- generation metrics: BLEU, ROUGE, METEOR, exact match
- regression metrics when numeric cost extraction succeeds
- classification metrics when predicted/reference labels are available
- rank-wise prediction dumping and merged reporting

## Inference

```bash
python3 GLM/scripts/inference.py \
  --config GLM/configs/runpod.yaml \
  --image path/to/car.jpg \
  --checkpoint outputs/run_name/last.pt
```

Optional JSON output:

```bash
python3 GLM/scripts/inference.py \
  --config GLM/configs/runpod.yaml \
  --image path/to/car.jpg \
  --checkpoint outputs/run_name/last.pt \
  --output outputs/inference/result.json
```

## Runpod Notes

This repository is structured so the `GLM/` directory can be cloned directly onto the GPU instance and used with minimal setup friction.

Recommended operational flow:

1. upload the cleaned dataset to a Hugging Face dataset repo
2. clone the project on Runpod
3. run `GLM/runpod_setup.sh`
4. confirm paths/configs
5. launch `GLM/runpod_ddp.sh`

## Metrics and Analysis

The evaluation stack is intentionally modular:

- `GLM/evaluation/prediction_schema.py`: normalized record format
- `GLM/evaluation/io.py`: rank-wise persistence and merging
- `GLM/evaluation/generation_metrics.py`: text-generation metrics
- `GLM/evaluation/regression_metrics.py`: cost-estimation metrics
- `GLM/evaluation/classification_metrics.py`: label metrics when available

This lets you keep training/eval orchestration thin while adding metrics without turning `evaluate.py` into a monolith.

## Current Status

The GLM pipeline now includes:

- dataset cleanup and split utilities
- Hugging Face dataset upload/download flow
- model loading for `GLM-4.6V-Flash`
- GLM chat-template based collator
- DDP-capable training loop with optional profiling
- evaluation and inference entrypoints
- WandB integration
- Runpod setup and launch scripts

## Remaining Practical Work Before a Full Run

- verify final dataset paths on the Runpod workspace
- run a 1-GPU smoke test end-to-end
- confirm memory behavior for your exact batch size / grad accumulation settings
- inspect training outputs and eval predictions before committing to a long run
