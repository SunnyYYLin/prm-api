import json
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from utils import query_llm, brief_info, sort_dict
import re

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}."
ID_KEY = 'idx'

with open('api_key.txt', 'r') as f:
    api_key = f.readlines()[0].strip()

client = OpenAI(api_key=api_key, base_url="https://zzzzapi.com/v1")

def question_key(name: str):
    match name:
        case 'math':
            return 'problem'
        case 'gsm8k':
            return 'question'
        case _:
            raise ValueError(f"Unknown dataset name: {name}")
        
def preprocess(data: list[dict[str, str]], data_name: str):
    for i, datum in enumerate(data):
        datum['idx'] = i
        datum['question'] = datum[question_key(data_name)]
        if question_key(data_name) != 'question':
            del datum[question_key(data_name)]
    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/sunnylin/projects/prm-api/data/math/train.jsonl')
    parser.add_argument('--dataset_name', type=str, default='math')
    parser.add_argument('--model', type=str, default='deepseek-v3')
    parser.add_argument('--num_processes', type=int, default=128)
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
        print(brief_info(data))
        data = preprocess(data, args.dataset_name)
        print(brief_info(data))
        
    # exit()
    
    output_path = dataset_path.with_name(dataset_path.stem + f'_{args.model}.jsonl')
    if not output_path.exists():
        output_path.touch()
    
    with open(output_path, 'r+') as f:
        results: list[dict[str, str]] = [json.loads(line) for line in f]
        tested_indices = set([result[ID_KEY] for result in results])
        print(f"Already Tested: {tested_indices}")
        
        def single_process(datum: dict[str, str]) -> dict|None:
            if datum[ID_KEY] in tested_indices:
                return datum
            user_prompt = datum['question']
            try:
                response = query_llm(SYSTEM_PROMPT, user_prompt, client)
            except Exception as e:
                print(f"Error: {e} at {datum[ID_KEY]}")
                return datum
            datum['reference'] = response
            return datum
        
        with Pool(args.num_processes) as p:
            for datum in tqdm(p.imap(single_process, data), total=len(data), 
                            desc='Querying LLM', dynamic_ncols=True):
                if datum.get('reference') is not None:
                    datum = sort_dict(datum, ID_KEY, 'question', 'reference')
                    results.append(datum)
                    f.write(json.dumps(datum) + '\n')
                else:
                    print(f"Skipping {datum[ID_KEY]}")