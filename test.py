import json
from utils import sort_dict

path = "/home/sunnylin/projects/prm-api/data/math_false_critiques.jsonl"
with open(path, 'r+') as f:
    results = [json.loads(line) for line in f]
    f.seek(0)
    for result in results:
        result['vr_score'] = 1
        sort_dict(result, 'index', 'prompt', 'completions', 'labels', 'vr_score')
        f.write(json.dumps(result) + '\n')
    f.truncate()
        
    