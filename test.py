import json
from utils import brief_info

path = "/home/sunnylin/projects/prm-api/data/math_VRs_rm.jsonl"

with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
    print(brief_info(results))

f = open('special.txt', 'w')
count = 0
for result in results:
    steps = result['response'].split('\n\n')
    if any(len(step)<5 for step in steps):
        f.write(f"Query:\n{result['query']}\n\n")
        f.write(f"Response:\n{result['response']}\n---\n")
        count += 1
f.close()
print(f"{count}/{len(results)}")