import json
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
import re
import time
import random
from pathlib import Path
from utils import query_llm, sort_dict, reduce_newlines

USER_TEMPLATE = "<problem>{problem}<\\problem>\n<solution>{solution}<\\solution>"

with open('api_key.txt', 'r') as f:
    api_key = f.readlines()[0].strip()

client = OpenAI(api_key=api_key, base_url="https://zzzzapi.com/v1")
    
def preprocess(dataset: list[dict], name: str) -> list[dict]:
    match name:
        case 'math':
            return _preprocess_math(dataset)
        case 'gsm8k':
            return _preprocess_gsk8k(dataset)
        case 'metamath':
            return _preprocess_metamath(dataset)
        case _:
            raise ValueError(f"Unknown dataset name: {name}")
        
def _preprocess_math(dataset: list[dict]) -> list[dict]:
    for datum in dataset:
        datum['id'] = datum['unique_id']
        datum['question'] = datum['problem']
        datum['gt_label'] = datum['answer']
        datum['answer'] = datum['solution']
        del(datum['problem'])
        del(datum['solution'])
        del(datum['subject'])
        del(datum['level'])
        del(datum['unique_id'])
        datum['dataset_name'] = 'math'
    return dataset

def _preprocess_gsk8k(dataset: list[dict]) -> list[dict]:
    for datum in dataset:
        datum['gt_label'] = datum['answer'].split('#### ')[-1]
        datum['answer'] = datum['answer'].removesuffix(f'#### {datum["gt_label"]}')
        datum['id'] = datum['idx']
        del(datum['idx'])
        datum['dataset_name'] = 'gsk8k'
    return dataset

def _preprocess_metamath(dataset: list[dict]) -> list[dict]:
    for i, datum in enumerate(dataset):
        datum['id'] = i
        datum['question'] = datum['query']
        datum['answer'] = datum['response']
        datum['dataset_name'] = 'metamath'
    return dataset

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser(description='Tag Math Critiques')
    parser.add_argument('--input_path', type=str, help='Path to the input file', \
        default="/home/sunnylin/projects/prm-api/data/math_false.jsonl")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', default='math')
    parser.add_argument('--num_processes', type=int, default=128, help='Number of processes to use')
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    with open(input_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
        dataset = preprocess(dataset, args.dataset_name)
    
    with open('./prompts/split_prompt.txt', 'r') as f:
        role_prompt = f.read()
    
    output_path = input_path.with_name(input_path.stem + '_steps.jsonl')
    if not output_path.exists():
        output_path.touch()
    with open(output_path, 'r') as f:
        split_data = [json.loads(line) for line in f]
        split_ids = set([datum['id'] for datum in split_data])
        print(f"Already Split: {split_ids}")

    f = open(output_path, 'a')
    
    def single_process(datum) -> dict|None:
        if datum['id'] in split_ids:
            return datum
        user_prompt = USER_TEMPLATE.format(problem=datum['question'], solution=datum['answer'])
        time.sleep(random.uniform(0, 1))
        try:
            steps = query_llm(role_prompt, user_prompt, client)
        except Exception as e:
            print(f"Error: {e} at {datum['id']}")
            return datum
        datum['steps'] = steps
        return datum
    
    with Pool(args.num_processes) as p:
        for datum in tqdm(p.imap(single_process, dataset), total=len(dataset), 
                           desc='Querying LLM', dynamic_ncols=True):
            if datum.get('steps') is not None:
                f.write(json.dumps(datum) + '\n')
            else:
                print(f"Skipping {datum['id']}")
            
    f.close()
    
    # postprocess
    with open(output_path, 'r+') as f:
        data = [json.loads(line) for line in f]
        f.seek(0)
        for datum in data:
            datum = sort_dict(datum, 'id', 'question', 'answer', 'gt_label', 'dataset_name')
            datum['steps'] = reduce_newlines(datum['steps'])
            f.write(json.dumps(datum) + '\n')
        f.truncate()
    print("Postprocessing Done!")