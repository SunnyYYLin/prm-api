import json

path = "/home/sunnylin/projects/prm-api/data/math_gsm8k_steps_1.jsonl"
with open(path, 'r+') as f:
    results = [json.loads(line) for line in f]
    results = [result for result in results if result['dataset_name'] == 'math']
    f.seek(0)
    for result in results:
        f.write(json.dumps(result) + '\n')
    f.truncate()
    