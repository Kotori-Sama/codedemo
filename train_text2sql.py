import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
# from guardrail.client import (
#     run_metrics,
#     run_simple_metrics,
#     create_dataset)

import json
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/train_config_text2sql.json')
parser.add_argument('--model_name', type=str, default='/new_disk/uni_group/jsr_temp/pre-trained-llms/llama2-7b-hf/')
parser.add_argument('--new_model', type=str, default='llama-2-7b-text2sql')
parser.add_argument('--dataset', type=str, default='./data/preprocessed_data/seq2seq_preprocessed_dataset.json')

parser=parser.parse_args()

print('加载配置文件')
with open(parser.config, 'r') as f:
    config=json.load(f)

if config is None:
    raise Exception("No config file found")

config['model_name']=parser.model_name
config['new_model']=parser.new_model
config['dataset']=parser.dataset

def load_model(model_name):

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config['bnb_4bit_compute_dtype'])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['use_4bit'],
        bnb_4bit_quant_type=config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['use_nested_quant'],
    )

    if compute_dtype == torch.float16 and config['use_4bit']:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    print("加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={'':torch.cuda.current_device()},
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        r=config['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("加载tokenizer")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

if __name__ == "__main__":
    model, tokenizer, peft_config = load_model(config['model_name'])
    print("加载数据集")
    data_files = {'train': './dataset/llama_preprocessed_train_dataset_natsql.json', 'test': './dataset/llama_preprocessed_test_dataset_natsql.json','dev': './dataset/llama_preprocessed_dev_dataset_natsql.json'}
    dataset = load_dataset('json', data_files='./data/preprocessed_data/seq2seq_preprocessed_dataset.json',split="train")
    dataset_shuffled = dataset.shuffle(seed=42)
    print(dataset[0])


    training_arguments = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optim'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        learning_rate=config['learning_rate'],
        fp16=config['fp16'],
        bf16=config['bf16'],
        max_grad_norm=config['max_grad_norm'],
        max_steps=config['max_steps'],
        warmup_ratio=config['warmup_ratio'],
        group_by_length=config['group_by_length'],
        lr_scheduler_type=config['lr_scheduler_type'],
        num_train_epochs=config['num_train_epochs'],
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config['packing'],
    )

    trainer.train()
    trainer.model.save_pretrained(config['output_dir'])
