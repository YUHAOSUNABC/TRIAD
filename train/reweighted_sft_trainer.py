import os
import logging
import torch
from typing import Dict, Any, List
from dataclasses import dataclass, field

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers import HfArgumentParser, TrainerCallback, AutoTokenizer
from datasets import load_dataset


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Reduce logging from other libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("trl").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"


@dataclass
class AdditionalArgs:
    """Additional arguments for training"""
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset (JSON format)"}
    )
    model_path: str = field(
        metadata={"help": "Path to the pretrained model"}
    )
    weight_field: str = field(
        default="decision_confidence",
        metadata={"help": "Field to use as confidence weight (decision_confidence or avg_confidence)"}
    )
    weight_min: float = field(
        default=0.1,
        metadata={"help": "Minimum weight value (to avoid zero weights)"}
    )
    weight_max: float = field(
        default=1.0,
        metadata={"help": "Maximum weight value"}
    )
    refuse_weight: float = field(
        default=1.0,
        metadata={"help": "Decision-based weight multiplier for Refuse samples (< 1.0 to down-weight). "
                          "Final weight = decision_weight * confidence_weight"}
    )


@dataclass
class WeightedDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that extends TRL's DataCollatorForLanguageModeling
    to preserve the 'weight' field for reweighted training.
    """

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract weights before calling parent collator
        weights = None
        if examples and "weight" in examples[0]:
            weights = torch.tensor([ex.pop("weight") for ex in examples], dtype=torch.float32)

        # Call parent collator for standard processing
        batch = super().torch_call(examples)

        # Add weights back to batch
        if weights is not None:
            batch["weight"] = weights

        return batch


class WeightedSFTTrainer(SFTTrainer):
    """
    SFT Trainer with sample reweighting based on confidence scores.

    Each sample's loss is multiplied by its weight (derived from confidence score).
    Higher confidence = higher weight = more influence on training.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss (token-level weighting).
        """
        # Extract weights from inputs (won't be passed to model)
        weights = inputs.pop("weight", None)

        # Get labels (keep in inputs for model forward)
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)

        # Compute loss manually to ensure weighted loss works correctly
        if labels is not None:
            logits = outputs.logits

            # Shift for causal LM (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute per-token loss (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Reshape to (batch_size, seq_len)
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Mask for valid tokens (labels != -100)
            mask = (shift_labels != -100).float()

            if weights is not None:
                # Expand sample weights to token level
                weights = weights.to(per_token_loss.device)
                token_weights = weights.unsqueeze(1).expand_as(mask)

                # Token-level weighted loss
                loss = (per_token_loss * mask * token_weights).sum() / (mask * token_weights).sum().clamp(min=1)
            else:
                # No weights: standard mean loss over valid tokens
                loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

            # Compute mean_token_accuracy (matches TRL's SFTTrainer behavior)
            with torch.no_grad():
                shift_predictions = shift_logits.argmax(dim=-1)
                correct = (shift_predictions == shift_labels) & (shift_labels != -100)
                correct_tokens = self.accelerator.gather_for_metrics(correct.sum())
                total_tokens = self.accelerator.gather_for_metrics(mask.sum().long())
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics["train"]["mean_token_accuracy"].append(accuracy)

            outputs.loss = loss
        else:
            # Fallback: use model's computed loss if available
            if outputs.loss is None:
                raise ValueError("Labels not provided and model did not compute loss")

        return (outputs.loss, outputs) if return_outputs else outputs.loss


class LoggingCallback(TrainerCallback):
    """Custom callback for minimal logging"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.global_step % 50 == 0:
            key_metrics = {k: v for k, v in logs.items()
                          if k in ['loss', 'learning_rate', 'epoch']}
            if key_metrics:
                logger.info(f"Step {state.global_step}: {key_metrics}")


def normalize_weights(weights: List[float], min_val: float, max_val: float) -> List[float]:
    """
    Normalize weights to [min_val, max_val] range.
    """
    if not weights:
        return weights

    w_min, w_max = min(weights), max(weights)

    if w_max == w_min:
        return [max_val] * len(weights)

    normalized = []
    for w in weights:
        norm_w = min_val + (w - w_min) / (w_max - w_min) * (max_val - min_val)
        normalized.append(norm_w)

    return normalized


def main():
    # Parse arguments
    parser = HfArgumentParser((SFTConfig, AdditionalArgs))
    training_args, additional_args = parser.parse_args_into_dataclasses()

    # Note: Wandb is configured via shell script using:
    # - WANDB_PROJECT env var
    # - --report_to wandb
    # - --run_name $RUN_NAME
    # The Trainer handles wandb initialization automatically

    # Print configuration
    logger.info("=" * 60)
    logger.info("Reweighted SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {additional_args.model_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Weight field: {additional_args.weight_field}")
    logger.info(f"Weight range: [{additional_args.weight_min}, {additional_args.weight_max}]")
    logger.info(f"Refuse weight: {additional_args.refuse_weight}")
    logger.info(f"Completion only loss: {training_args.completion_only_loss}")
    logger.info("=" * 60)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        additional_args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=additional_args.dataset_path,
    )["train"]

    logger.info(f"Dataset size: {len(dataset)}")

    # Extract and normalize weights
    logger.info("Processing sample weights...")
    weight_field = additional_args.weight_field

    # Step 1: Confidence-based weights (normalized to [weight_min, weight_max])
    if weight_field in dataset.column_names:
        raw_weights = dataset[weight_field]
        confidence_weights = normalize_weights(
            raw_weights,
            additional_args.weight_min,
            additional_args.weight_max
        )
        logger.info(f"Confidence weights ({weight_field}) - min: {min(confidence_weights):.4f}, "
                   f"max: {max(confidence_weights):.4f}, "
                   f"mean: {sum(confidence_weights)/len(confidence_weights):.4f}")
    else:
        logger.warning(f"Weight field '{weight_field}' not found. Using uniform confidence weights.")
        confidence_weights = [1.0] * len(dataset)

    # Step 2: Decision-based weights (refuse_weight for Refuse samples, 1.0 for others)
    refuse_w = additional_args.refuse_weight
    if "decision" in dataset.column_names and refuse_w != 1.0:
        decisions = dataset["decision"]
        decision_weights = [refuse_w if d == "refuse" else 1.0 for d in decisions]

        from collections import Counter
        dec_counts = Counter(decisions)
        logger.info(f"Decision distribution: {dict(dec_counts)}")
        logger.info(f"Decision weights - Refuse: {refuse_w}, Others: 1.0")
    else:
        decision_weights = [1.0] * len(dataset)
        if refuse_w != 1.0:
            logger.warning("'decision' column not found in dataset. Ignoring refuse_weight.")

    # Step 3: Combined weight = decision_weight * confidence_weight
    combined_weights = [d * c for d, c in zip(decision_weights, confidence_weights)]
    dataset = dataset.add_column("weight", combined_weights)

    logger.info(f"Final weights - min: {min(combined_weights):.4f}, "
               f"max: {max(combined_weights):.4f}, "
               f"mean: {sum(combined_weights)/len(combined_weights):.4f}")

    # Remove columns not needed for training
    columns_to_remove = [col for col in ["avg_confidence", "decision_confidence", "decision"]
                         if col in dataset.column_names]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    # Disable thinking mode for Qwen3.5 (official per-sample approach)
    dataset = dataset.add_column(
        "chat_template_kwargs",
        [{"enable_thinking": False} for _ in range(len(dataset))]
    )
    logger.info("Added per-sample chat_template_kwargs: enable_thinking=False")

    # Create weighted data collator
    data_collator = WeightedDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=training_args.completion_only_loss,
    )

    # Initialize trainer
    # TRL automatically handles:
    # - Prompt-completion data format detection
    # - Chat template application
    # - completion_only_loss masking (labels=-100 for prompt tokens)
    # WeightedSFTTrainer adds sample-level loss weighting
    logger.info("Initializing WeightedSFTTrainer...")
    trainer = WeightedSFTTrainer(
        args=training_args,
        model=additional_args.model_path,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
