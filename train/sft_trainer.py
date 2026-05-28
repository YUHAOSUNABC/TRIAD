import os
import logging

from trl import SFTConfig, SFTTrainer
from transformers import HfArgumentParser, TrainerCallback, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass, field

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


class LoggingCallback(TrainerCallback):
    """Custom callback for minimal logging"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.global_step % 50 == 0:
            key_metrics = {k: v for k, v in logs.items()
                          if k in ['loss', 'learning_rate', 'epoch']}
            if key_metrics:
                logger.info(f"Step {state.global_step}: {key_metrics}")


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
    logger.info("SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {additional_args.model_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Epochs: {training_args.num_train_epochs} | LR: {training_args.learning_rate}")
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

    # Load dataset (TRL will handle preprocessing for conversational data)
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=additional_args.dataset_path,
    )["train"]

    logger.info(f"Dataset size: {len(dataset)}")

    # Disable thinking mode for Qwen3.5 (official per-sample approach)
    dataset = dataset.add_column(
        "chat_template_kwargs",
        [{"enable_thinking": False} for _ in range(len(dataset))]
    )
    logger.info("Added per-sample chat_template_kwargs: enable_thinking=False")

    # Initialize trainer
    # TRL automatically handles:
    # - Prompt-completion data format detection
    # - Chat template application
    # - completion_only_loss masking (labels=-100 for prompt tokens)
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        args=training_args,
        model=additional_args.model_path,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[LoggingCallback()],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()