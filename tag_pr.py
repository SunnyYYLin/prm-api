import json
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
import re
import time
import random
from pathlib import Path
from utils import query_llm, brief_info

NUM_PROCESSES = 1
SYSTEM_PROMPT = "You are a helpful assistant"

with open('api_key.txt', 'r') as f:
    api_key = f.readlines()[2].strip()

# client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
# client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
client = OpenAI(api_key=api_key, base_url="https://zzzzapi.com/v1")
    
def tag_response(response: str):
    steps = response.split('\n\n')
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    return tagged_response

def construct_critique_prompt(result: dict, template: str):
    problem = result['query']
    # gt_cot = result['gt_cot']
    tagged_response = tag_response(result['response'])
    score = 'correct' if result['offset'] < 0 else 'incorrect'
    critique = template.format(problem=problem, tagged_response=tagged_response, score=score)
    return critique

def extract_critique(critique: str):
    analysis_sections = re.findall(r'<analysis_\d+>.*?</analysis_\d+>', critique, re.DOTALL)
    conclusion_section = re.search(r'<conclusion>.*?</conclusion>', critique, re.DOTALL)
    analysis_tags = [int(re.search(r'\\boxed{(-?\+?\d+)}', section).group(1)) for section in analysis_sections]
    conclusion_tag = re.search(r'\\boxed{(-?\+?\d+)}', conclusion_section.group(0)).group(1) if conclusion_section else None
    return analysis_tags, conclusion_tag

if __name__ == '__main__': 
    input_path = Path('test500_qwen25-math-cot_-1_seed0_t0.7_s0_e-1_prm_800k_offset.jsonl')
    with open(input_path, 'r') as f:
        results = [json.loads(line) for line in f]
        print(brief_info(results))
    
    with open('./prompts/tag_prompt.txt', 'r') as f:
        template = f.read()
        
    def single_process(result) -> dict|None:
        if result['hacking_idx'] in critiqued_indices:
            return result
        time.sleep(random.uniform(0, 1))
        prompt = construct_critique_prompt(result, template)
        try:
            critique = query_llm(SYSTEM_PROMPT, prompt, client)
            result['critique_tags'], result['conclusion_tag'] = extract_critique(critique)
        except Exception as e:
            print(f"Error: {e} at {result['hacking_idx']}")
            result['critique_tags'], result['conclusion_tag'] = None, None
            result['critique'] = critique
            return result
        result['critique'] = critique
        return result
    
    output_path = input_path.with_name(input_path.stem + '_critiques.jsonl')
    if not output_path.exists():
        output_path.touch()
    with open(output_path, 'r+') as f:
        critiqued = [json.loads(line) for line in f]
        critiqued_indices = set([critique['hacking_idx'] for critique in critiqued])
        print(f"Already Critiqued: {critiqued_indices}")
    
        with Pool(NUM_PROCESSES) as p:
            for result in tqdm(p.imap(single_process, results), total=len(results), 
                            desc='Tagging Process Rewards', dynamic_ncols=True):
                if result.get('critique') is not None:
                    f.write(json.dumps(result) + '\n')
                else:
                    print(f"Skipping {result['hacking_idx']}")
                
        f.close()