import os
import re
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
from tqdm import tqdm

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.json')
    parser.add_argument('--model_name', type=str, default='./results/checkpoint-1750')
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

    model.config.use_cache = True
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

def get_labels(question_db,ans,INST=False):

    inst=question_db.split('|')
    table_names=[]
    column_names_list=[]
    for table in inst[1:]:
        tables_and_colunms=table.split(':')
        table_name=tables_and_colunms[0]
        column_names=tables_and_colunms[1].split(',')
        table_names.append(table_name)
        column_names_list.append(column_names)
    
    resp=ans.split('|')
    if INST:
        resp=resp[:-1]
    predict_table_names=[]
    predict_column_names_list=[]
    for table in resp:
        tables_and_colunms=table.split(':')
        table_name=tables_and_colunms[0]
        if len(tables_and_colunms) == 1:
            predict_table_names.append(table_name)
            predict_column_names_list.append([])
            continue
        column_names=[]
        for i in tables_and_colunms[1].split(','):
            if i != '':
                column_names.append(i.split('.')[1])

        #column_names=[i.split('.')[1] for i in tables_and_colunms[1].split(',')]
        # print(tables_and_colunms[1].split(','))
        # print(column_names)
        predict_table_names.append(table_name)
        predict_column_names_list.append(column_names)

    predict_table_labels=[]
    predict_column_names_list_labels=[]

    for i in range(len(table_names)):
        table_name=table_names[i]

        if table_name in predict_table_names:
            predict_table_labels.append(1)
        else:
            predict_table_labels.append(0)
        
        predict_column_names_labels=[]
        for j in range(len(column_names_list[i])):
            column_name=column_names_list[i][j]
            flag=False
            for predict_column_names in predict_column_names_list:
                if column_name in predict_column_names:
                    flag=True
                    break
            if flag:
                predict_column_names_labels.append(1)    
            else:
                predict_column_names_labels.append(0)
        predict_column_names_list_labels.append(predict_column_names_labels)
        
    return predict_table_labels,predict_column_names_list_labels


def gen_txt(model, tokenizer, prompt, temp=0.7, max_length=500):
  
    # Suppress Hugging Face pipeline logging
    logging.set_verbosity(logging.CRITICAL )
    # logger = logging.get_logger("logs")
    # logger.info("INFO")
    # logger.warning("WARN")
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

    return generated_text



if __name__ == '__main__':
    
    model, tokenizer ,peft_config = load_model(config['model_name'])

    dataset = json.load(open(config['dev_dataset'],'r'))
    #prompt='What are all distinct countries where singers above age 20 are from?|stadium:stadium id,location,name,capacity,highest,lowest,average|singer:singer id,name,country,song name,song release year,age,is male|concert:concert id,concert name,theme,stadium id,year|singer in concert:concert id,singer id'
    #text_gen_eval_wrapper(model,tokenizer,prompt,show_metrics=False)
    tr_table_labels=[]
    tr_column_labels=[]
    pre_table_labels=[]
    pre_column_labels=[]
    i=0
    instructions = []
    responses = []
    for data in tqdm(dataset, desc="Processing"):
        # i+=1
        # if i == 5:
        #      break
        text=data['text']
        pattern = r'<s>\[INST\] (.*?)\[/INST\] (.*?)</s>'
        matches = re.search(pattern, text)
        question_db = matches.group(1)
        instructions.append("<s>[INST] %s [/INST]"%question_db)
        ans = matches.group(2)

        true_table_labels,true_column_names_list_labels=get_labels(question_db,ans)
    

    # pipe = pipeline(task="text-generation",
    #                 model=model,
    #                 tokenizer=tokenizer,
    #                 max_length=500,
    #                 do_sample=True,
    #                 temperature=0.0)

    # Generate text using the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=500,
                    temperature=0.0)
    # result = pipe(f"<s>[INST] {prompt} [/INST]")
    # generated_text = result[0]['generated_text']
    print(pipe(instructions[0]))
    generated_texts = []
    for i in tqdm(range(len(instructions)), desc="generating"):
        instruction = instructions[i]
        generated_texts.append(pipe(instruction)[0]['generated_text'])
        print(generated_texts)
    print(generated_texts)
    for generated_text in tqdm(generated_texts, desc="Processing"):
        
        #print(question_db)
        # generated_texts=gen_txt(model,tokenizer,instructions)
        #print(generated_text)  
        pattern = r'\[INST\] (.*?) \[/INST\] (.*?)\[/INST\]'
        #pattern = r'\[INST\](.*?)\[/INST\](.*?)</s>'
        matches = re.search(pattern, generated_text)

        INST=False
        if matches:
            question_db = matches.group(1)
            ans = matches.group(2)
        else:
            #i+=1
            #print(f'输出格式错误，可能会导致出错的个数:{i}')
            pattern = r'\[INST\] (.*?) \[/INST\] (.*?)'
            matches = re.search(pattern, generated_text)
            question_db = matches.group(1)
            ans = matches.group(2)
            INST=True

        responses.append(ans)

        predict_table_labels,predict_column_names_list_labels=get_labels(question_db,ans,INST=INST)

        tr_table_labels.append(true_table_labels)
        tr_column_labels+=true_column_names_list_labels
        pre_table_labels.append(predict_table_labels)
        pre_column_labels+=predict_column_names_list_labels

    for ins, ans in zip(instructions, responses):
        print(ins)
        print(ans)
        print("")
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.preprocessing import MultiLabelBinarizer
    log=''
    #print(tr_table_labels,pre_table_labels)
    mlb = MultiLabelBinarizer()
    tr_table_labels_b = mlb.fit_transform(tr_table_labels)
    pre_table_labels_b = mlb.fit_transform(pre_table_labels)
    accuracy = accuracy_score(tr_table_labels_b, pre_table_labels_b)
    print(f"表分类准确率：{accuracy:.2f}")
    recall = recall_score(tr_table_labels_b, pre_table_labels_b, average='micro')
    print(f"表分类召回率：{recall:.2f}")
    log+=f"表分类准确率：{accuracy:.2f}\n"+f"表分类召回率：{recall:.2f}\n"

    #print(tr_column_labels,pre_column_labels)

    mlb = MultiLabelBinarizer()
    tr_column_labels_b = mlb.fit_transform(tr_column_labels)
    pre_column_labels_b = mlb.fit_transform(pre_column_labels)
    #print(tr_column_labels_b,pre_column_labels_b)

    accuracy = accuracy_score(tr_column_labels_b, pre_column_labels_b)
    print(f"列分类准确率：{accuracy:.2f}")
    recall = recall_score(tr_column_labels_b, pre_column_labels_b, average='micro')
    print(f"列分类召回率：{recall:.2f}")
    log+=f"列分类准确率：{accuracy:.2f}\n"+f"列分类召回率：{recall:.2f}\n"

    with open('./log/evaluate.log','w') as f:
        f.write(log)