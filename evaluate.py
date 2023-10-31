import os
import re
import sys
import torch
import logging 
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline, 
)
#from sklearn.metrics import accuracy_score, recall_score
#from sklearn.preprocessing import MultiLabelBinarizer
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import json
import argparse
from tqdm import tqdm

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.json')
    parser.add_argument('--model_name', type=str, default='./results/checkpoint-17760')
    parser.add_argument('--dev_dataset', type=str, default='./dataset/llama_preprocessed_dev_dataset_natsql.json')
    
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

def get_labels(question_db,ans):

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
    # if INST:
    #     resp=resp[:-1]
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
            col_inf=i.split('.')
            if i == '' or len(col_inf)==1:
                continue
            else:
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
        predict_column_names_list_labels+=predict_column_names_labels
        
    return predict_table_labels,predict_column_names_list_labels


def gen_txt(model, tokenizer, prompt, temp=0, max_length=500):
  
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

def evaluate(true_labels,pre_labels):

    assert len(true_labels)==len(pre_labels)
    length=len(true_labels)
    TP=0
    FP=0
    FN=0
    for i in range(length):
        true_bit=true_labels[i]
        pre_bit=pre_labels[i]
        if true_bit==1 and pre_bit==1:
            TP+=1
        elif true_bit==1 and pre_bit==0:
            FN+=1
        elif true_bit==0 and pre_bit==1:
            FP+=1
    
    if TP+FP!=0:
        accuracy=TP/(TP+FP)
    else:
        accuracy=0
        
    if TP+FN!=0:
        recall=TP/(TP+FN)
    else:
        recall=0

    return accuracy,recall

def set_logger(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler('%s/evaluate.log' % (log_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    #consoleHandler = logging.StreamHandler(sys.stdout)
    #consoleHandler.setFormatter(logFormatter)
    #logger.addHandler(consoleHandler)
    return logger

if __name__ == '__main__':
    if os.path.exists('./log/prompt_and_response.log'):
        os.remove('./log/prompt_and_response.log')
    logger=set_logger('./log')
    
    model, tokenizer ,peft_config = load_model(config['model_name'])

    dataset = json.load(open(config['dev_dataset'],'r'))
    #prompt='What are all distinct countries where singers above age 20 are from?|stadium:stadium id,location,name,capacity,highest,lowest,average|singer:singer id,name,country,song name,song release year,age,is male|concert:concert id,concert name,theme,stadium id,year|singer in concert:concert id,singer id'
    #text_gen_eval_wrapper(model,tokenizer,prompt,show_metrics=False)
    # tr_table_labels=[]
    # tr_column_labels=[]
    # pre_table_labels=[]
    # pre_column_labels=[]
    i=0
    fail=0
    responses = []
    table_accuracy_list=[]
    table_recall_list=[]
    col_accuracy_list=[]
    col_recall_list=[]
    for data in tqdm(dataset, desc="Processing"):
        # i+=1
        # if i == 5:
        #     break
        text=data['text']
        pattern = r'<s>\[INST\] (.*?)\[/INST\] (.*?)</s>'
        matches = re.search(pattern, text)
        question_db = matches.group(1)
        ans = matches.group(2)

        true_table_labels,true_column_names_list_labels=get_labels(question_db,ans)
        logger.info(f'问题和数据库格式:{question_db}')
        logger.info(f'真值：{ans}')
        
        
        #print(question_db)
        generated_text=gen_txt(model,tokenizer,question_db)
        f_write={'text':text,
                 'generated_text':generated_text,
                 'question_db': question_db,
                 'true_table_labels':true_table_labels,
                 'true_column_labels':true_column_names_list_labels}
        #print(generated_text)  
        pattern = r'\[INST\] (.*?) \[/INST\] (.*?)\[/INST\]'
        #pattern = r'\[INST\](.*?)\[/INST\](.*?)</s>'
        matches = re.search(pattern, generated_text)

        #INST=False
        if matches:
            question_db = matches.group(1)
            ans = matches.group(2)
        else:
            #i+=1
            logger.info(f'输出格式错误')
            pattern = r'\[INST\] (.*?) \[/INST\] (.*)'
            matches = re.search(pattern, generated_text)
            question_db = matches.group(1)
            ans = matches.group(2)
            #INST=True
        
        f_write['response']=ans
        
        #print(ans)
        
        try: 
            predict_table_labels,predict_column_names_list_labels=get_labels(question_db,ans)
            f_write['predict_table_labels']=predict_table_labels
            f_write['predict_column_labels']=predict_column_names_list_labels
            logger.info(f'预测：{ans}')
            logger.info(f'真值表标签：{true_table_labels}\t真值列标签：{true_column_names_list_labels}')
            logger.info(f'预测表标签：{predict_table_labels}\t预测列标签：{predict_column_names_list_labels}')
        except IndexError:
            fail+=1
            logger.info(f'输出格式错误，可能会导致出错的个数:{fail}')
            continue

        with open('./log/prompt_and_response.log','a') as f:
            json.dump(f_write,f,ensure_ascii=False,indent=4)
            f.write('\n')

        accuracy,recall=evaluate(true_table_labels,predict_table_labels)
        table_accuracy_list.append(accuracy)
        table_recall_list.append(recall)
        logger.info(f'表分类准确率：{accuracy:.2f}\t表分类召回率：{recall:.2f}')


        accuracy,recall=evaluate(true_column_names_list_labels,predict_column_names_list_labels)
        col_accuracy_list.append(accuracy)
        col_recall_list.append(recall)
        logger.info(f'列分类准确率：{accuracy:.2f}\t列分类召回率：{recall:.2f}')

        logger.info('================================================================================')


    table_accuracy=sum(table_accuracy_list)/len(table_accuracy_list)
    table_recall=sum(table_recall_list)/len(table_recall_list)
    col_accuracy=sum(col_accuracy_list)/len(col_accuracy_list)
    col_recall=sum(col_recall_list)/len(col_recall_list)
    logger.info(f'表分类准确率：{table_accuracy*100:.2f}%\t表分类召回率：{table_recall*100:.2f}%')
    logger.info(f'列分类准确率：{col_accuracy*100:.2f}%\t列分类召回率：{col_recall*100:.2f}%')





    
    


    




