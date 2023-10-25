import json
import argparse

def parse_option():
    parser = argparse.ArgumentParser("")
    

    parser.add_argument('--input_path', type = str, default = "./data/preprocessed_data/preprocessed_train_spider.json", 
                        help = '''
                            options:
                                ./data/spider/train_spider.json
                                ./data/spider/dev.json
                            ''')
    parser.add_argument('--output_path', type = str, default = "./data/preprocessed_data/llama_preprocessed_dataset.json", 
                        help = "the filepath of preprocessed dataset.")


    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt=parse_option()

    llama_preprocessed_dataset = []    

    with open(opt.input_path, "r") as f:
        preprocessed_dataset = json.load(f)

        for preprocessed_data in preprocessed_dataset:
            llama_preprocessed_data={}

            # 构造instruction
            db_schemas=preprocessed_data["db_schema"]
            question=preprocessed_data["question"]
            processed_tables=[]
            for db_schema in db_schemas:
                tabel_name=db_schema["table_name"]
                column_names=db_schema["column_names"]
                processed_table=tabel_name+':'+','.join(column_names)
                processed_tables.append(processed_table)
            instruction = f"[INST] {question+'|'+'|'.join(processed_tables)}"
            #print(instruction)

            #构造response
            table_labels=preprocessed_data["table_labels"]
            column_labels=preprocessed_data["column_labels"]
            processed_res_tables=[]
            for index in range(len(table_labels)):
                table = table_labels[index]
                columns = column_labels[index]
                if table == 1:
                    db_schema = db_schemas[index]
                    tabel_name = db_schema["table_name"]
                    column_names = db_schema["column_names"]
                    processed_columns=[]
                    for j in range(len(columns)):
                        column=columns[j]
                        if column == 1:
                            column_name=tabel_name + '.' + column_names[j]   
                            processed_columns.append(column_name)
                    processed_res_tables.append(tabel_name+':'+','.join(processed_columns))
                    
            response = f"[/INST] {'|'.join(processed_res_tables)}"
            prompt = instruction + response
            llama_preprocessed_data['text']='<s>'+prompt+'</s>'
            llama_preprocessed_dataset.append(llama_preprocessed_data)

        
    
    with open(opt.output_path, 'w') as f:
        llama_preprocessed_data_str=json.dumps(llama_preprocessed_dataset, indent = 2, ensure_ascii = False)
        f.write(llama_preprocessed_data_str)








