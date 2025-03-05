import json
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from utils import query_llm, brief_info, sort_dict
import re
from grader import math_equal
from parser import extract_answer, parse_ground_truth, choice_answer_clean, parse_question
from tqdm import tqdm

STEP_SEP = '\n\n'
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}."

with open('api_key.txt', 'r') as f:
    api_key = f.readlines()[0].strip()

client = OpenAI(api_key=api_key, base_url="https://zzzzapi.com/v1")
ID_KEY = 'idx'

def preprocess(data: list[dict[str, str]], data_name: str):
    for i, datum in enumerate(data):
        datum['idx'] = i
        datum['question'] = parse_question(datum, data_name)
        _, datum['gt'] = parse_ground_truth(datum, data_name)
    return data

def is_multi_choice(answer):
    if answer is None:
        return False  # 或者 return some default value
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def evaluate(results: list[dict[str, str]], data_name: str):
    CHOICES_ALPHA = ['A', 'B', 'C', 'D', 'E']
    for result in tqdm(results, desc='Evaluating', dynamic_ncols=True):
        result['pred'] = extract_answer(result['response'], data_name)
        
        if result['gt'] in CHOICES_ALPHA and result['pred'] not in CHOICES_ALPHA:
            result['pred'] = choice_answer_clean(result['pred'])
        elif is_multi_choice(result['gt']) and not is_multi_choice(result['pred']):
            if isinstance(result['pred'], (tuple, list)):
                result['pred'] = ''.join(map(str, result['pred']))
            elif not isinstance(result['pred'], str):
                result['pred'] = str(result['pred'])
            result['pred'] = ''.join([c for c in result['pred'] if c in CHOICES_ALPHA])
        
        result['score'] = math_equal(result['pred'], result['gt'])
    return results

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser(description='Tag Math Critiques')
    parser.add_argument('--dataset', type=str, help='Path to the dataset', \
        default="./data/minerva_math/test.jsonl")
    parser.add_argument('--data_name', type=str, help='Name of the dataset', \
        default="minerva_math")
    parser.add_argument('--model', type=str, default="o1-mini-2024-09-12", help='Model to use')
    parser.add_argument('--num_processes', type=int, default=128, help='Number of processes to use')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
        print(brief_info(data))
        data = preprocess(data, args.data_name)
        print(brief_info(data))
    # exit()
    
    with open('./prompts/tag_prompt.txt', 'r') as f:
        template = f.read()
    
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
            datum['response'] = response
            return datum
        
        with Pool(args.num_processes) as p:
            for datum in tqdm(p.imap(single_process, data), total=len(data), 
                            desc='Querying LLM', dynamic_ncols=True):
                if datum.get('response') is not None:
                    datum = sort_dict(datum, ID_KEY, 'question', 'response')
                    results.append(datum)
                    f.write(json.dumps(datum) + '\n')
                else:
                    print(f"Skipping {datum[ID_KEY]}")
        
    print(f'Results:\n{brief_info(results)}')
    results = evaluate(results, args.data_name)
    print(f'Evaluation Results:\n{brief_info(results)}')
    print(f'Correct Rate: {sum([result["score"] for result in results]) / len(results)}')
    output_path = output_path.with_name(output_path.stem + '_eval.jsonl')
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_path}")
    
