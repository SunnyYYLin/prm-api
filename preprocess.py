import json
from utils import sort_dict
from utils import brief_info

path = "/home/sunnylin/projects/prm-api/data/gsm8k-DPO-001-PRM-filter-epo0_hacking.jsonl"
with open(path, 'r+') as f:
    results = [json.loads(line) for line in f.readlines()]
    print(brief_info(results))

f = open('data/unformatted.txt', 'w')
count = 0
for i, result in enumerate(results):
    steps = result['response'].split('\n\n')
    if any(len(step)<4 for step in steps):
        print(result['idx'])
        f.write(f"Query:\n{result['query']}\n\n")
        f.write(f"Response:\n{result['response']}\n---\n")
        count += 1
        del(results[i])
f.close()

with open(path, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
print(f"{count}/{len(results)}")