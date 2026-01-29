# Insurance Damage Assessment AI

An end-to-end vision-language AI system that automatically generates professional damage assessment reports from vehicle images. The system fine-tunes LLaVA-NeXT using LoRA for efficient domain adaptation to the insurance industry.

## Overview

Insurance adjusters manually assess vehicle damage from photos, which is time-consuming and inconsistent. This system takes a single image of a damaged vehicle and generates a detailed report including:

- Damage type (dent, scratch, crack, etc.)
- Damage location on the vehicle
- Severity assessment
- Estimated repair cost

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LABELLING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Images + Metadata                                          │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐    ┌──────────────┐                          │
│   │   GPT-4o     │    │  Qwen2.5-VL  │   Auto-captioning        │
│   └──────────────┘    └──────────────┘                          │
│          │                   │                                   │
│          └─────────┬─────────┘                                   │
│                    ▼                                             │
│          ┌──────────────────┐                                    │
│          │   GPT-4o-mini    │   Quality evaluation               │
│          └──────────────────┘                                    │
│                    │                                             │
│                    ▼                                             │
│          High-Quality Captions                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐                                          │
│   │ Dataset Cleanup  │   Remove metadata for image-only input   │
│   └──────────────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │ Train/Test Split │   90/10 split                            │
│   └──────────────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │  LLaVA Fine-tune │   LoRA + 4-bit quantization              │
│   └──────────────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │  HuggingFace Hub │   Model deployment                       │
│   └──────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Insurance/
├── data_labelling/
│   ├── gpt_captioning/          # GPT-4o caption generation
│   ├── qwen_captioning/         # Qwen2.5-VL caption generation
│   └── evaluation_and_correction/  # Quality filtering with GPT-4o-mini
│
├── llava/
│   ├── config/
│   │   └── config.yaml          # Training hyperparameters
│   ├── dataset/
│   │   ├── train.json           # Training data
│   │   └── test.json            # Test data
│   ├── scripts/
│   │   ├── train.py             # Main training entry point
│   │   ├── lightning_module.py  # PyTorch Lightning module
│   │   ├── model_loader.py      # Model loading with quantization
│   │   ├── prepare_data.py      # Dataset and data collator
│   │   ├── train_test_split.py  # Data splitting utility
│   │   └── dataset_cleanup.py   # Remove metadata from prompts
│   └── tests/                   # Unit tests
│
└── google_drive/                # Data download utilities
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Insurance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Remove metadata from dataset for image-only training:

```bash
cd llava/scripts

# Preview changes without saving
python dataset_cleanup.py ../dataset/train.json ../dataset/train_cleaned.json --dry-run

# Clean training data
python dataset_cleanup.py ../dataset/train.json ../dataset/train_cleaned.json

# Clean test data
python dataset_cleanup.py ../dataset/test.json ../dataset/test_cleaned.json
```

### 2. Training

```bash
cd llava/scripts
python train.py
```

Training configuration can be modified in `llava/config/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| model_id | llava-hf/llava-v1.6-mistral-7b-hf | Base model |
| batch_size | 8 | Training batch size |
| max_epochs | 10 | Maximum training epochs |
| learning_rate | 1e-4 | Learning rate |
| lora_r | 16 | LoRA rank |
| lora_alpha | 32 | LoRA alpha |
| use_4bit | true | Enable 4-bit quantization |

### 3. Inference

After training, the model is pushed to HuggingFace Hub at `Raghav77/Insurance_Adjuster`.

## Technical Details

### Model Architecture

- **Base Model**: LLaVA-NeXT (llava-v1.6-mistral-7b-hf)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Quantization**: 4-bit BitsAndBytes for reduced memory footprint
- **Parameters**: ~7B total, ~10M trainable with LoRA

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| ROUGE-1 | Unigram overlap between generated and reference text |
| ROUGE-L | Longest common subsequence similarity |
| BertScore F1 | Semantic similarity using DeBERTa embeddings |

### Data Format

Input conversation format (after cleanup):

```json
{
  "id": "000775",
  "image": "000775.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "You are an Insurance Adjuster. Evaluate the car damage shown in the image and provide a detailed report including damage type, location, severity, and estimated repair cost."
    },
    {
      "role": "assistant",
      "content": "The vehicle in the image shows a dent located on the side panel..."
    }
  ]
}
```

## Technologies Used

| Category | Technologies |
|----------|-------------|
| Vision-Language Models | LLaVA-NeXT, Qwen2.5-VL, GPT-4o |
| Training Framework | PyTorch Lightning, PEFT, Accelerate |
| Quantization | BitsAndBytes |
| Experiment Tracking | Weights & Biases |
| Model Hosting | HuggingFace Hub |
| Data Processing | Pandas, PIL, NLTK |

## Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | 660 |
| Test | 74 |
| Total | 734 |

## Future Improvements

- Expand training dataset to 2,000-5,000 samples for better generalization
- Add more damage categories (shattered glass, tire damage, etc.)
- Integrate object detection model for precise damage localization
- Implement confidence scores for cost estimates
- Add support for multiple images per assessment

## License

[Add your license here]

## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) for the base vision-language model
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library
- [PyTorch Lightning](https://lightning.ai/) for the training framework