import argparse
import json
import sqlite3
from collections import Counter
import os


def find_values(cursor, table_name):
    try:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [column[1] for column in cursor.fetchall()]
        most_common_values = {}
        for column in columns:
            cursor.execute(
                f"SELECT {column} FROM {table_name} GROUP BY {column} ORDER BY COUNT(*) DESC;"
            )
            result = cursor.fetchall()
            if len(result)>0:
                most_common_values[column] = [result[0][0]]
            if len(result)>1:
                most_common_values[column].append(result[1][0])
            if len(result)>2:
                most_common_values[column].append(result[2][0])

        return most_common_values

    except sqlite3.Error as e:
        print(f"Error reading from table {table_name}: {e}")
        return None


def get_tuples(db_file):
    ans=[]
    try:
        # 连接到 SQLite 数据库
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()
        # 获取数据库中的表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        # 逐表读取每列中出现最多的数据
        for table in tables:
            table_name = table[0]
            most_common_values = find_values(cursor, table_name)
            for column in most_common_values:
                values=most_common_values[column]
                ans+=[f"{table_name}.{column} -> {value}" if not isinstance(value, str) 
                      else f"{table_name}.{column} -> '{value}'"  
                      for value in values]
        
    except sqlite3.Error as e:
        print(f"Error reading from database {db_file}: {e}")
    finally:
        if connection:
            connection.close()
        return ans


def get_scheam(table,column):
    
    ans=[]
    for table_index in range(len(table)):
        cols=[col[1] for col in column if col[0]==table_index]
        table_name=table[table_index]
        ans.append(f'{table_name}({",".join(cols)})')
    return ans

def get_fk(table,column,foreign_keys):
    ans=[]
    for fk in foreign_keys:
        source=column[fk[0]][-1]
        source_table=table[column[fk[0]][0]]
        target=column[fk[-1]][-1]
        target_table=table[column[fk[-1]][0]]
        ans.append(f'{source_table}.{source} -> {target_table}.{target}')
    return ans


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default='./data/train_spider.json',help='path of train_spider.json')
    parser.add_argument('--tables', type=str,default='./data/tables.json',help='path of tables.json')
    parser.add_argument('--output_dir', type=str,default='./data/db_scheme.json',help='path of output file')
    args = parser.parse_args()

    sample_sql_path=args.train_data
    tables_path=args.tables

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
                    'column_names_original':db['column_names_original'],
                    "str_fk_list":get_fk(db['table_names_original'],db['column_names_original'],db['foreign_keys']),
                    "str_tuples":get_tuples(f'./spider/database/{db_id}/{db_id}.sqlite')
                }

        with open(args.output_dir,'w',encoding='utf-8') as f3:
            json.dump(db_scheme,f3,ensure_ascii=False,indent=4)