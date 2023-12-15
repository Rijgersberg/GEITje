import datasets
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    AutoModelForCausalLM, AutoTokenizer

from util import pack


def train(base_model, context_length, dataset_name, new_model_name):
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    # fix padding (mostly for inference, later for finetuning changed to unk_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # data
    dataset = datasets.load_dataset(dataset_name)

    # it is customary to train LLMs by fully "packing" the context length with
    # fragments of one or more documents
    packed_train_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={'dataset': dataset['train'],
                    'tokenizer': tokenizer,
                    'context_length': context_length})

    packed_validation_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={'dataset': dataset['validation'],
                    'tokenizer': tokenizer,
                    'context_length': context_length})

    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    training_steps = 10_000_000_000 // (torch.cuda.device_count() * per_device_train_batch_size *
                                       gradient_accumulation_steps * context_length)

    save_steps = training_steps // (6 * 4) + 1
    eval_steps = training_steps // (6 * 8) + 1
    # training
    training_args = TrainingArguments(
        max_steps=training_steps,
        optim='adamw_bnb_8bit',
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        warmup_steps=int(training_steps * 0.1),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        evaluation_strategy='steps',
        eval_steps=eval_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        eval_accumulation_steps=gradient_accumulation_steps,
        save_strategy='steps',
        save_steps=save_steps,
        bf16=True,
        output_dir='/tmp/geitje/output',
        report_to=["tensorboard", 'wandb'],
        logging_steps=1,
        logging_first_step=True,
        hub_model_id=new_model_name,
        hub_private_repo=True,
        push_to_hub=True,
        hub_strategy='all_checkpoints',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_train_dataset,
        eval_dataset=packed_validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == '__main__':
    train(
        base_model='mistralai/Mistral-7B-v0.1',
        context_length=8192,
        dataset_name='Rijgersberg/GEITJE-pretrain-10b',
        new_model_name='Rijgersberg/GEITje-7B',
    )