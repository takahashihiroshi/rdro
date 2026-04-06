import os
import random

import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.kto import KTOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from trainers import RDROTrainer

if __name__ == "__main__":
    # Hyperparameters
    base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path = "trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness"
    alpha = 0.39
    seed = 42

    num_train_epochs = 1
    learning_rate = 5e-7
    per_device_train_batch_size = 16

    # Unique key: {algorithm}-{base_model}-{dataset}-{alpha}-{seed}
    unique_key = f"RDRO-Llama-8B-UF-G-{alpha}-{seed}"

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
    )

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        token=os.environ.get("HF_TOKEN"),
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Training
    train_dataset = datasets.load_dataset(path=dataset_path, split="train")

    config = KTOConfig(
        output_dir=f"logs/{unique_key}",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        seed=seed,
        remove_unused_columns=False,
        report_to="tensorboard",
        use_liger_kernel=False,
        precompute_ref_log_probs=False,
    )

    # Trainer
    trainer = RDROTrainer(
        alpha=alpha,
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Training
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    # Save model
    trainer.save_model(f"models/{unique_key}")
