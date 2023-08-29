import pandas as pd
import textwrap


def eval(str):
    return [s.strip(" '[]") for s in str.split(",")]

data = pd.read_csv("fine_tuning/0000-1000.csv")

NEWLINE = '\n'

def create_work(id, abstract, keywords, openai):
    diff = keyword_diff(keywords, openai)

    all_keywords = sorted(set(eval(keywords) + diff))
    all_keywords = ["# " + k if k in diff else k for k in all_keywords]

    return textwrap.dedent(f"""
{"=" * 80}

{id}

{NEWLINE.join(textwrap.wrap(abstract, width=80))}

§
{NEWLINE.join(all_keywords)}
§

[n]   
""")


def keyword_diff(keywords, openai):
    keywords = [l.lower() for l in eval(keywords)]
    openai = [l.lower() for l in eval(openai)]
    return sorted(set(openai) - set(keywords))


comp_data = pd.read_csv("fine_tuning/openai_100.csv")

m = data.merge(comp_data, on="id", how="inner")

with open("fine_tuning/tagging.txt", "w") as f:
    for id, abstract, keywords, openai in zip(m.id, m.abstract, m.concepts, m.openai):
        f.write(create_work(id, abstract, keywords, openai))

