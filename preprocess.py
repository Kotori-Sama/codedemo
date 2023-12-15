from sql_metadata import Parser
import re
def white_space_fix(s):
    modified_query = re.sub(r'(\(|\))', r' \1 ', s)
    return modified_query

def replace_consecutive_spaces(input_str):
    # 使用正则表达式替换连续的空格为单个空格
    output_str = re.sub(r'\s+', ' ', input_str)
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
            alias_lower = "t{}".format(i)
            alias_upper = "T{}".format(i)

            if alias_lower in tables_aliases:
                new_tables_aliases[alias_lower] = tables_aliases[alias_lower]
            elif alias_upper in tables_aliases:
                new_tables_aliases[alias_upper] = tables_aliases[alias_upper]
        
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
w_data = []

with jsonlines.open(jsonl_file_path, 'r') as jsonl_file:
    for item in jsonl_file:
        w = {}
        w['instruction'] = item['instruction']
        w['input'] = ''
        w['output'] = remove_table_alias(add_asc(lower(replace_consecutive_spaces(white_space_fix(item['output'])))))
        w_data.append(w)

with jsonlines.open(output_file_path, 'w') as merged_jsonl_file:
    for item in w_data:
        merged_jsonl_file.write(item)