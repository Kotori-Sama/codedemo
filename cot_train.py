import json
################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

PROMPT={

    'from':'Analyzing the composition of the from clause, based on the problem information, we should choose tables ',
    'from_condition':',and join them on ',#slot: ON

    'select':'Analyzing the composition of the select clause, the problem shows that the user wants to query information for columns ',
    'select_aggregation':['There should be an aggregation function for ',' on column '], # slot: name of aggregation function
    'select_no_aggregation':'No aggregation function is required on column ',
    'select_distinct':'Meanwhile, the user wants to query distinct values of column.',
    # select needs '.' 
    'where':'Analyzing the composition of the where clause, the problem shows that the limiting conditions should be: ',
    'where_nested_query':['column ','requires nested queries',', and a nested query statement ',' needs to be added.'],#slot:column name ; nested query statement
    'where_no_nested_query':['column',' does not require nested queries'],#slot: column name
    'where_condition':',and the condition is',
    'where_no_condition':'Analyzing the composition of the where clause, the problem shows that no limiting conditions are required.',
    # where needs '.' 
    'group_by':'Analyzing the composition of the group by clause:',
    'group_by_column':['The problem shows that column ',' should be used for grouping.'],#slot: column name
    'group_by_no_column':'The problem shows that no columns should be used for grouping.',
    'group_by_condition':'And the condition is ',#slot: Having

    'order_by':'Analyzing the composition of the order by clause:',
    'order_by_column':['The problem shows that column ',' should be used for sorting,and the order is '], #slot:column name ; order
    'order_by_no_column':'The problem shows that no columns should be used for sorting.',

    'limit':'Analyzing the composition of the limit clause:',
    'limit_number':'the problem shows that there is need to limit the number of results,and the number is ',#slot: number
    'limit_no_number':'the problem shows that there is no need to limit the number of results.',

    'intersect':['The problem shows that we should intersect two queries.\nConsidering the first query:\n','Considering the second query:\n'],

    'except':['The problem shows that we should exclude the second query from the first query.\nConsidering the first query:\n','Considering the second query:\n'],

    'union':['The problem shows that we should union two queries.\nConsidering the first query:\n','Considering the second query:\n'],

    'ans':'So, the final output SQL statement is:',#slot: gold spl

    'OpenAI':['### Complete sqlite SQL query only and with no explanation\n### SQLite SQL tables , with their properties:\n#\n','#\n### '],

    'nested':['In addition,the problem shows that the nested query is required.Considering the nested query:','\n'],
}

def random_sample(path = './random_sample.sql'):
    import openpyxl
    import random
    workbook = openpyxl.load_workbook('./Spider榜单bad case分析.xlsx')
    sheet = workbook.active  # 或者可以通过名称选择工作表：sheet = workbook['Sheet1']
    ans=[]
    for row in sheet.iter_rows():
        gold_sql=row[2].value
        gold_db_id=row[3].value
        gold_question=row[4].value
        with open('./dev.json','r',encoding='utf-8') as f:
            all_data=json.load(f)
            for data in all_data:
                if data['question']==gold_question:
                    ans.append({'query':gold_sql,'db_id':gold_db_id,'question':gold_question,'sql':data['sql']})
                    break

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(random.sample(ans,10),f,ensure_ascii=False, indent=4)
    workbook.close()
    return ans,path

def parse_col_unit(col_unit,table_names_original,column_names_original,need_table_name=True):
    """
        这里忽视了非null情况
    """
    #print(col_unit)
    agg_id=col_unit[0]
    col_id=col_unit[1]
    isDistinct=col_unit[2]
    if isDistinct:
        ans='DISTINCT '
    else:
        ans=''
    # print(column_names_original)
    col_name=column_names_original[col_id][1]
    t_index=column_names_original[col_id][0]
    t_name=table_names_original[t_index]+'.' if need_table_name else ''
    ans+=AGG_OPS[agg_id]+'('+t_name + col_name+')' if agg_id!=0 else t_name + col_name
    return ans

def parse_select_tables(from_,table_names_original):
    table_name=[]
    table_units=from_['table_units']
    for table_unit in table_units:
        if table_unit[0]=='table_unit': # 嵌套查询怎么办？
            table_name.append(table_names_original[table_unit[1]])
    return table_name

def parse_condition(conds,table_names_original,column_names_original,need_table_name=True):
    #多复用函数，谨慎修改
    # print(conds)
    nested_query=None
    ans=[]
    length=len(conds)
    if length==0:
        return [],None
    for cond_unit in conds:

        if cond_unit in COND_OPS:
            ans.append(cond_unit) 
            continue

        if cond_unit[0]:
            not_op='not '
        else:
            not_op=''

        col_name1=parse_col_unit(cond_unit[2][1],table_names_original,column_names_original,need_table_name=need_table_name)
        if type(cond_unit[3])==list:
            col_name2=parse_col_unit(cond_unit[3],table_names_original,column_names_original,need_table_name=need_table_name)
        elif type(cond_unit[3])==dict:
            col_name2=extract_sql_str(cond_unit[3],{'table_names_original':table_names_original,
                                              'column_names_original':column_names_original})
            nested_query=(col_name2,cond_unit[3])
        else:
            col_name2=cond_unit[3]
        str=f'{col_name1} {not_op}{WHERE_OPS[cond_unit[1]]} {col_name2}'
        ans.append(str)
    return ans,nested_query

def parse_from_condition(from_,table_names_original,column_names_original):
    return parse_condition(from_['conds'],table_names_original,column_names_original)

def parse_select_columns(select,table_names_original,column_names_original):
    isDistinct=select[0]
    aggregation=[]
    col_names=[]
    for (agg_id, val_unit) in select[1]:
        col_unit=val_unit[1]
        col_name=parse_col_unit(col_unit,table_names_original,column_names_original,need_table_name=False)
        col_names.append(col_name)
        aggregation.append(AGG_OPS[agg_id]) if agg_id!=0 else aggregation.append('')
        #print(agg_id, val_unit)
    
    if len(col_names)==1 and col_names[0]=='*' and aggregation[0]=='':
        col_names=[]
        aggregation=[]
        for col in column_names_original[1:]:
            col_names.append(col[1])
            aggregation.append('')

    return isDistinct,col_names,aggregation
        
def parse_where_condition(where,table_names_original,column_names_original):
    
    return parse_condition(where,table_names_original,column_names_original)
    
def parse_groupBy(groupBy,table_names_original,column_names_original):
    col_names=[]
    for col_unit in groupBy:
        col_name=parse_col_unit(col_unit,table_names_original,column_names_original,need_table_name=False)
        col_names.append(col_name)
    return col_names

def parse_having(having,table_names_original,column_names_original):
    return parse_condition(having,table_names_original,column_names_original,need_table_name=False)

def parse_orderBy(orderBy,table_names_original,column_names_original):
    if len(orderBy)==0:
        return [],''
    order = orderBy[0]
    ans=[]
    for val_unit in orderBy[1]:
        col_unit=val_unit[1]
        col_name=parse_col_unit(col_unit,table_names_original,column_names_original,need_table_name=False)
        ans.append(col_name)
    return ans,order

def parse_limit(limit):
    if limit == None:
        return 0
    else:
        return limit

def extract_sql(sql,db_scheme,having_intersect=False,having_except=False,having_union=False):
    #print(sql['query'])
    final_ans=''

    skeleton=sql['sql']
    query=sql['query']

    select=skeleton['select']
    from_=skeleton['from']
    where=skeleton['where']
    groupBy=skeleton['groupBy']
    orderBy=skeleton['orderBy']
    having=skeleton['having']
    limit=skeleton['limit']
    intersect=skeleton['intersect']
    except_=skeleton['except']
    union=skeleton['union']

    table_names_original=db_scheme['table_names_original']
    column_names_original=db_scheme['column_names_original']

    #print(intersect)

    if intersect is not None and not having_intersect:
        ans1=extract_sql(sql,db_scheme,having_intersect=True)
        ans2=extract_sql({'sql':intersect,'query':query},db_scheme,having_intersect=True)
        final_ans+=ans1.join(PROMPT['intersect'])+ans2
        return final_ans+PROMPT['ans']+query

    if except_ is not None and not having_except:
        ans1=extract_sql(sql,db_scheme,having_except=True)
        ans2=extract_sql({'sql':except_,'query':query},db_scheme,having_except=True)
        final_ans+=ans1.join(PROMPT['except'])+ans2
        return final_ans+PROMPT['ans']+query
    
    if union is not None and not having_union:
        ans1=extract_sql(sql,db_scheme,having_union=True)
        ans2=extract_sql({'sql':union,'query':query},db_scheme,having_union=True)
        final_ans+=ans1.join(PROMPT['union'])+ans2
        return final_ans+PROMPT['ans']+query


    select_tables=parse_select_tables(from_,table_names_original)
    from_condition,nested_query=parse_from_condition(from_,table_names_original,column_names_original)
    if len(from_condition):
        final_ans+=PROMPT['from']+",".join(select_tables)+PROMPT['from_condition']+" ".join(from_condition)+'.\n'
    else:
        final_ans+=PROMPT['from']+",".join(select_tables)+'.\n'
    #print(final_ans)

    isDistinct,col_names,aggregation=parse_select_columns(select,table_names_original,column_names_original)
    final_ans+=PROMPT['select']+",".join(col_names)+'.'

    agg_flag=False
    for index in range(len(col_names)):
        if aggregation[index]!='':
            agg_flag=True
            final_ans+=aggregation[index].join(PROMPT['select_aggregation'])+col_names[index]+'.'
        else:
            final_ans+=PROMPT['select_no_aggregation']+col_names[index]+'.'
    
    # if not agg_flag:
    #     final_ans+=PROMPT['select_no_aggregation']

    if isDistinct:
        final_ans+=PROMPT['select_distinct']
    final_ans+='\n'

    #print(final_ans)

    where_condition,nested_query=parse_where_condition(where,table_names_original,column_names_original)
    
    if len(where_condition):
        final_ans+=PROMPT['where']+' '.join(where_condition)+'.\n'
    else:
        final_ans+=PROMPT['where_no_condition']+'\n'

    #print(final_ans)

    group_by_info=parse_groupBy(groupBy,table_names_original,column_names_original)
    if len(group_by_info):
        final_ans+=PROMPT['group_by']+','.join(group_by_info).join(PROMPT['group_by_column'])
        having_info,nested_query=parse_having(having,table_names_original,column_names_original)
        if len(having_info):
            final_ans+=PROMPT['group_by_condition']+' '.join(having_info)+'.'
    else:
        final_ans+=PROMPT['group_by']+PROMPT['group_by_no_column']
    final_ans+='\n'
    
    #print(final_ans)

    order_by_info,order=parse_orderBy(orderBy,table_names_original,column_names_original)
    if len(order_by_info):
        final_ans+=PROMPT['order_by']+','.join(order_by_info).join(PROMPT['order_by_column'])+order.upper()+'.\n'
    else:
        final_ans+=PROMPT['order_by']+PROMPT['order_by_no_column']+'\n'
    
    #print(final_ans)

    limit_info=parse_limit(limit)
    if limit_info:
        final_ans+=PROMPT['limit']+PROMPT['limit_number']+str(limit_info)+'.'
    else:
        final_ans+=PROMPT['limit']+PROMPT['limit_no_number']
    final_ans+='\n'

    #print(final_ans)

    if having_except or having_intersect or having_union :
        return final_ans
    
    if nested_query is not None:
        nested_query_str,nested_query=nested_query
        nested_query_str=nested_query_str[1:-1]
        nested_query:dict
        if nested_query.get('sql') is not None:
            final_ans+=nested_query_str.join(PROMPT['nested'])+extract_sql(nested_query,db_scheme)
        else:
            final_ans+=nested_query_str.join(PROMPT['nested'])+extract_sql({'sql':nested_query,'query':query},db_scheme)
        #print(final_ans)
    
    return final_ans+PROMPT['ans']+query

def extract_sql_str(sql,db_scheme,having_intersect=False,having_except=False,having_union=False):
    #print('='*10)
    final_ans=''

    try :
        skeleton=sql['sql']
    except:
        skeleton=sql
    #query=sql['query']

    select=skeleton['select']
    from_=skeleton['from']
    where=skeleton['where']
    groupBy=skeleton['groupBy']
    orderBy=skeleton['orderBy']
    having=skeleton['having']
    limit=skeleton['limit']
    intersect=skeleton['intersect']
    except_=skeleton['except']
    union=skeleton['union']

    table_names_original=db_scheme['table_names_original']
    column_names_original=db_scheme['column_names_original']


    if intersect is not None and not having_intersect:
        ans1=extract_sql_str(sql,db_scheme,having_intersect=True)
        ans2=extract_sql_str({'sql':intersect},db_scheme,having_intersect=True)
        final_ans+=ans1+' intersect '+ans2
        return final_ans.join(['(',')'])

    if except_ is not None and not having_except:
        ans1=extract_sql_str(sql,db_scheme,having_except=True)
        ans2=extract_sql_str({'sql':except_},db_scheme,having_except=True)
        final_ans+=ans1+' except '+ans2
        return final_ans.join(['(',')'])
    
    if union is not None and not having_union:
        ans1=extract_sql_str(sql,db_scheme,having_union=True)
        ans2=extract_sql_str({'sql':union},db_scheme,having_union=True)
        final_ans+=ans1+' union '+ ans2
        return final_ans.join(['(',')'])


    select_tables=parse_select_tables(from_,table_names_original)
    from_condition,nested_query=parse_from_condition(from_,table_names_original,column_names_original)
    isDistinct,col_names,aggregation=parse_select_columns(select,table_names_original,column_names_original)
    SELECT_str='select '
    if isDistinct:
        SELECT_str+='distinct '
    for index in range(len(col_names)):
        if aggregation[index]!='':
            col_names[index]=aggregation[index]+'('+col_names[index]+')'
    final_ans+='select '+",".join(col_names)+' from '+",".join(select_tables)
    if len(from_condition):
        final_ans+=' on '+' '.join(from_condition)
    
    #print(final_ans)

    where_condition,nested_query=parse_where_condition(where,table_names_original,column_names_original)
    
    if len(where_condition):
        final_ans+=' where '+' '.join(where_condition)

    #print(final_ans)

    group_by_info=parse_groupBy(groupBy,table_names_original,column_names_original)
    if len(group_by_info):
        final_ans+='group by '+','.join(group_by_info)
        having_info,nested_query=parse_having(having,table_names_original,column_names_original)
        if len(having_info):
            final_ans+=' having '+' '.join(having_info)
    
    #print(final_ans)

    order_by_info,order=parse_orderBy(orderBy,table_names_original,column_names_original)
    if len(order_by_info):
        final_ans+=' order by '+','.join(order_by_info)+' '+order.upper()
    
    #print(final_ans)

    limit_info=parse_limit(limit)
    if limit_info:
        final_ans+=' limit '+str(limit_info)+' '
    

    # print(final_ans)

    if having_except or having_intersect or having_union :
        return final_ans
    
    return final_ans.join(['(',')'])

def openai_prompt(db_scheme_strs,question):
    prompt_db_scheme=''
    for db_str in db_scheme_strs:
        prompt_db_scheme+='# '+db_str+'\n'
    return prompt_db_scheme.join(PROMPT['OpenAI'])+question+'\n\n'
    

if __name__ == '__main__':
    db_scheme_path='./data/spider/db_scheme.json'
    train_data_path='./data/spider/train_spider.json'
    with open(train_data_path, 'r', encoding='utf-8') as f1 :
        train_data=json.load(f1)
    with open(db_scheme_path, 'r', encoding='utf-8') as f2 :
        db_scheme=json.load(f2)
    ans=[]
    for data in train_data:
        promt_sql=extract_sql(data,db_scheme=db_scheme[data['db_id']])
        db_scheme_strs=db_scheme[data['db_id']]['str_list']
        question=data['question']
        openai_p=openai_prompt(db_scheme_strs,question)
        ans.append({'text':openai_p+promt_sql})
    
    with open('./data/train.list','w',encoding='utf-8') as f:
        json.dump(ans,f,ensure_ascii=False,indent=4)

    















                
            

