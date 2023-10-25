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
import json
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.json')
    parser.add_argument('--model_name', type=str, default='./results/checkpoint-950')
    parser.add_argument('--dev_dataset', type=str, default='/home/fintech/jnz/codedemo/dataset/llama_preprocessed_dev_dataset_natsql.json')
    
    opt=parser.parse_args()
    return opt

parser=get_opt()
print('加载配置文件')
with open(parser.config, 'r') as f:
    config=json.load(f)

if config is None:
    raise Exception("No config file found")

config['model_name']=parser.model_name
config['dev_dataset']=parser.dev_dataset

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
        device_map=config['device_map'],
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

def text_gen_eval_wrapper(model, tokenizer, prompt, model_id=1, show_metrics=True, temp=0.7, max_length=200):
    """
    A wrapper function for inferencing, evaluating, and logging text generation pipeline.

    Parameters:
        model (str or object): The model name or the initialized text generation model.
        tokenizer (str or object): The tokenizer name or the initialized tokenizer for the model.
        prompt (str): The input prompt text for text generation.
        model_id (int, optional): An identifier for the model. Defaults to 1.
        show_metrics (bool, optional): Whether to calculate and show evaluation metrics.
                                       Defaults to True.
        max_length (int, optional): The maximum length of the generated text sequence.
                                    Defaults to 200.

    Returns:
        generated_text (str): The generated text by the model.
        metrics (dict): Evaluation metrics for the generated text (if show_metrics is True).
    """
    # Suppress Hugging Face pipeline logging
    logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temp)

    # Generate text using the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]['generated_text']

    # Find the index of "### Assistant" in the generated text
    index = generated_text.find("[/INST] ")
    if index != -1:
        # Extract the substring after "### Assistant"
        substring_after_assistant = generated_text[index + len("[/INST] "):].strip()
    else:
        # If "### Assistant" is not found, use the entire generated text
        substring_after_assistant = generated_text.strip()

    if show_metrics:
        # Calculate evaluation metrics
        pass
    else:
        pass

if __name__ == '__main__':
    
    model, tokenizer ,peft_config= load_model(config['model_name'])


    




