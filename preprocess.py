from sql_metadata import Parser
import re
from collections import Counter
def has_double_quotes_between_select_and_where(sql_query):

    # 找到 SELECT 和 WHERE 的索引
    select_index = sql_query.find('select')
    where_index = sql_query.find('where')

    # 如果找到了 SELECT 和 WHERE
    if select_index != -1 and where_index != -1:
        # 检查两者之间的部分是否包含双引号
        portion_between_select_and_where = sql_query[select_index + len('select'):where_index].strip()
        return '"' in portion_between_select_and_where
    else:
        return False


def find_duplicate_values(my_dict):
    # 创建一个计数器，统计每个值出现的次数
    value_counts = Counter(my_dict.values())

    # 从计数器中找出重复的值
    duplicate_values = {value for value, count in value_counts.items() if count > 1}

    # 找出拥有重复值的关键字
    duplicate_keys = {key for key, value in my_dict.items() if value in duplicate_values}

    for key in duplicate_keys:
        my_dict.pop(key, None)
    
    return my_dict

def white_space_fix(s):
    modified_query = re.sub(r'(\(|\))', r' \1 ', s)
    return modified_query
# remove ";"
def remove_semicolon(s):
    if s.endswith(";"):
        s = s[:-1]
    return s

# double quotation -> single quotation 
def double2single(s):
    return s.replace("\"", "'")

def replace_consecutive_spaces_except_quotes(input_str):
    # 使用正则表达式替换除引号内部的连续空格为单个空格
    output_str = re.sub(r'\s+(?=(?:(?:[^"\']*["\']){2})*[^"\']*$)', ' ', input_str)
    return output_str

# convert everything except text between single quotation marks to lower case
def lower(s):
    in_quotation = False
    out_s = ""
    for char in s:
        if in_quotation:
            out_s += char
        else:
            out_s += char.lower()
        
        if char == "'" or char == "\"":
            if in_quotation:
                in_quotation = False
            else:
                in_quotation = True
    
    return out_s

def add_asc(s):
    pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
    if "order by" in s and "asc" not in s and "desc" not in s:
        for p_str in pattern.findall(s):
            s = s.replace(p_str, p_str + " asc")

    return s

def remove_table_alias(s):
    try:
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            alias = "t{}".format(i)
            if alias in tables_aliases:
                new_tables_aliases[alias] = tables_aliases[alias]
        new_tables_aliases = find_duplicate_values(new_tables_aliases)
        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)
    
        return s
    except:
        return s
    
import jsonlines

# 输入输出的JSONL文件路径
jsonl_file_path = ''
output_file_path = ''

#分割instruction中creat语句， 例如：'--Using valid SQLite, answer the following'
separate_flag = ''
w_data = []

with jsonlines.open(jsonl_file_path, 'r') as jsonl_file:
    for item in jsonl_file:
        w = {}
        temp_ins = item['instruction']
        temp_ins_replace = temp_ins.split(separate_flag)[0]
        temp_ins = temp_ins.replace(temp_ins.split(separate_flag)[0], lower(temp_ins_replace))
        w['instruction'] = temp_ins
        w['input'] = ''
        w['output'] = lower(item['output'])
        if has_double_quotes_between_select_and_where(w['output']):
            w['output'] = remove_table_alias(add_asc(replace_consecutive_spaces_except_quotes(white_space_fix(remove_semicolon(w['output'])))))
        else:
            w['output'] = double2single(remove_table_alias(add_asc(replace_consecutive_spaces_except_quotes(white_space_fix(remove_semicolon(w['output']))))))
        w_data.append(w)

with jsonlines.open(output_file_path, 'w') as merged_jsonl_file:
    for item in w_data:
        merged_jsonl_file.write(item)