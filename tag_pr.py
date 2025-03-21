import json
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from utils import query_llm, brief_info
import re

SYSTEM_PROMPT = "You are a helpful assistant"

with open('api_key.txt', 'r') as f:
    api_key = f.readlines()[0].strip()

client = OpenAI(api_key=api_key, base_url="https://zzzzapi.com/v1")

def preprocess(data: list[dict[str, ]]):
    for i, datum in enumerate(data):
        datum['index'] = datum['idx']
        datum['prompt'] = datum['query']
        datum['completions'] = datum['response'].split('\n\n')
        del(datum['query'])
        del(datum['idx'])
        del(datum['min_reward'])
        del(datum['response'])
    return data

def construct_critique_prompt(result: dict, template: str):
    critique = template.replace('<gt_cot>', result['gt_cot'])
    completions = ''
    for i, completion in enumerate(result['completions']):
        completions += f"<step_{i+1}>\n{completion}\n<\\step_{i+1}>\n\n"
    critique = critique.replace('<completions>', completions)
    critique = critique.replace('<prompt>', result['prompt'])
    return critique

def extract_critique(critique: str):
    critique = critique.removeprefix('```json').removesuffix('```') # Remove code block
    critique = re.sub(r'\\(?![\\/bfnrt])', r'\\\\', critique)
    critique = json.loads(critique)
    labels: list[int] = []
    analysis = []
    for v in critique.values():
        if v['judgement'] == 'correct':
            labels.append(1)
        elif v['judgement'] == 'incorrect':
            labels.append(-1)
        else:
            labels.append(0)
        analysis.append(v['analysis'])
    return labels, analysis

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser(description='Tag Math Critiques')
    parser.add_argument('--input_path', type=str, help='Path to the input file', \
        default="/home/sunnylin/projects/prm-api/data/gsm8k-DPO-001-PRM-filter-epo0_hacking.jsonl")
    parser.add_argument('--num_processes', type=int, default=128, help='Number of processes to use')
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    with open(input_path, 'r') as f:
        results = [json.loads(line) for line in f]
        print(brief_info(results))
        results = preprocess(results)
        print(brief_info(results))
    
    with open('./prompts/tag_prompt.txt', 'r') as f:
        template = f.read()
    
    output_path = input_path.with_name(input_path.stem + '_critiques.jsonl')
    if not output_path.exists():
        output_path.touch()
    with open(output_path, 'r+') as f:
        critiqued = [json.loads(line) for line in f]
        critiqued_indices = set([critique['index'] for critique in critiqued])
        print(f"Already Critiqued: {critiqued_indices}")
        
        def single_process(result) -> dict|None:
            if result['index'] in critiqued_indices:
                return result
            user_prompt = construct_critique_prompt(result, template)
            # print(user_prompt)
            try:
                critique = query_llm(SYSTEM_PROMPT, user_prompt, client)
                labels, analysis = extract_critique(critique)
                assert len(labels) == len(result['completions']), \
                    f"Length Mismatch: {len(labels)} vs {len(result['completions'])}"
            except Exception as e:
                print(f"Error: {e} at {result['index']}")
                # if len(result['completions']) == 2:
                    # print(user_prompt)
                    # print(critique)
                return result
            result['labels'] = labels
            result['analysis'] = analysis
            return result
    
        with Pool(args.num_processes) as p:
            for result in tqdm(p.imap(single_process, results), total=len(results), 
                            desc='Tagging Process Rewards', dynamic_ncols=True):
                if result.get('labels') is not None:
                    del(result['gt_cot'])
                    f.write(json.dumps(result) + '\n')
                else:
                    print(f"Skipping {result['index']}")
                
        f.close()