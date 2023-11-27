import os
import openai
import json
from tqdm import tqdm
OPENAI_API_KEY='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'
openai.api_key=OPENAI_API_KEY
# completion = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )
def generate_txt(text):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
      #{"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
      {"role": "user", "content": f'{text}'}
    ],
    temperature=0
  )
    return completion.choices[0].message['content']

def get_text(path='./test_cases.txt'):
    with open(path, 'r') as f:
        content = f.read()
        return [x.strip() for x in content.split('========================================')  ]

if __name__ == "__main__":
    all_txt=get_text()
    if os.path.exists('./COT+GPT4.txt'):
        os.remove('./COT+GPT4.txt')
    with open('./COT+GPT4.txt', 'a') as f:
        for txt in tqdm(all_txt):
            gen_txt=generate_txt(txt) 
            #print(gen_txt)
            json.dump({'txt':txt,'generate_txt':gen_txt}, f,ensure_ascii=False,indent=4)
            
            #print(i)

        