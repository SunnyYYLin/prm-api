import re
from openai import OpenAI

def query_llm(role_prompt: str, user_prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    steps = response.choices[0].message.content
    return steps

def sort_dict(dictionary: dict, *sorted_keys):
    sort_fn = lambda x: sorted_keys.index(x[0]) if x[0] in sorted_keys else 25565
    return dict(sorted(dictionary.items(), key=sort_fn))

def reduce_newlines(text: str):
    raw_text = repr(text)[1:-1]
    raw_text = re.sub(r'\\+n\\+n(\\n|\\\\n)*', '\n\n', raw_text)
    reduced_text = raw_text.encode().decode('unicode_escape')
    return reduced_text

def brief_info(data):
    info: dict[str, int|list] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            info[k] = brief_info(v)
        return info
    elif isinstance(data, list):
        return (len(data), brief_info(data[0])) if len(data) > 0 else (len(data), )
    else:
        return str(type(data))