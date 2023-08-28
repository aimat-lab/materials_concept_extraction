import pandas as pd
import textwrap


def eval(str):
    return [s.strip(" '[]") for s in str.split(",")]

data = pd.read_csv("fine_tuning/0000-1000.csv")

NEWLINE = '\n'

def create_work(id, abstract, keywords):
    return textwrap.dedent(f"""
{"=" * 80}

{id}

{NEWLINE.join(textwrap.wrap(abstract, width=80))}

§
{NEWLINE.join(sorted(eval(keywords)))}
§

[n]   
""")

def create_tags(id, keywords):
    return textwrap.dedent(f"""
{"=" * 80}

{id}

§
{NEWLINE.join(sorted(eval(keywords)))}
§  
""")

with open("fine_tuning/tagging.txt", "w") as f:
    for id, abstract, keywords in zip(data.id, data.abstract, data.concepts):
        f.write(create_work(id, abstract, keywords))

comp_data = pd.read_csv("fine_tuning/openai_100.csv")

with open("fine_tuning/compare.txt", "w") as f:
    for id, keywords in zip(comp_data.id, comp_data.openai):
        f.write(create_tags(id, keywords))
