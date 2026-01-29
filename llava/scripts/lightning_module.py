import lightning as L
import torch

from llava.scripts.utils import DamageReportMetrics


class LlavaPLModule(L.LightningModule):
    def __init__(self, config, processor, model, max_length: int):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("model", "processor"))
        self.config = config
        self.processor = processor
        self.model = model
        self.max_length = max_length
        self.learning_rate = config.get("lr", 1e-4)
        self.metrics = DamageReportMetrics()

    def forward(self, **inputs):
        return self.model(**inputs)

    def _split_batch(self, batch):
        answers = batch.pop("answers", None)
        return batch, answers

    def training_step(self, batch, batch_idx):
        model_inputs, _ = self._split_batch(batch)
        outputs = self.forward(**model_inputs)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_inputs, answers = self._split_batch(batch)
        generation_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}

        generated_ids = self.model.generate(
            **generation_inputs,
            max_new_tokens=self.max_length,
        )

        input_ids = model_inputs.get("input_ids")
        if input_ids is not None:
            generated_ids = generated_ids[:, input_ids.size(1) :]

        predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        answers = answers or [""] * len(predictions)

        # Compute semantic metrics
        predictions_clean = [pred.strip() for pred in predictions]
        answers_clean = [ans.strip() for ans in answers]

        metrics = self.metrics.compute(predictions=predictions_clean, references=answers_clean)

        # Log all metrics
        self.log("val_rouge1", metrics['rouge1'], prog_bar=False, sync_dist=True)
        self.log("val_rougeL", metrics['rougeL'], prog_bar=False, sync_dist=True)
        self.log("val_bertscore_f1", metrics['bertscore_f1'], prog_bar=True, sync_dist=True)

        return metrics['bertscore_f1']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
