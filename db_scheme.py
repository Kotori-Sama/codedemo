import json
sample_sql_path='./data/spider/train_spider.json'
tables_path='./spider/tables.json'
def get_scheam(table,column):
    
    ans=[]
    for table_index in range(len(table)):
        cols=[col[1] for col in column if col[0]==table_index]
        table_name=table[table_index]
        ans.append(f'{table_name}({",".join(cols)})')
    return ans

with open(sample_sql_path, 'r', encoding='utf-8') as f1 :
    f2=open(tables_path,'r',encoding='utf-8')
    sample_sql = json.load(f1)
    tables = json.load(f2)
    db_id_list=[sql['db_id'] for sql in sample_sql]
    db_scheme={}
    for db in tables:
        db_id=db['db_id']
        if db_id in db_id_list:
            db_scheme[db_id]={
                "str_list":get_scheam(db['table_names_original'],db['column_names_original']),
                'table_names_original':db['table_names_original'],
                'column_names_original':db['column_names_original']
            }

    with open('./data/spider/db_scheme.json','w',encoding='utf-8') as f3:
        json.dump(db_scheme,f3,ensure_ascii=False,indent=4)