import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

between_brackets = re.compile(r"\[([^]]+)\]")  # extract text between square brackets


def keep_first_valid(text):
    return [line.strip() for line in text.split("\n") if line.strip()][0]


def add_last_bracket(text: str):
    if "]" not in text:
        return text[: text.rindex(",")] + "]"  # add last bracket after last comma
    else:
        return text


def trim_to_first_close_bracket(text: str):
    return text[: text.index("]") + 1]  # trim to first close bracket


def _eval(text):
    return [
        concept.strip()
        for concept in text.replace("'", "").split(",")
        if concept.strip()
    ]


def process_work(text):
    try:
        text = keep_first_valid(text)
        text = add_last_bracket(text)
        text = trim_to_first_close_bracket(text)
        text = between_brackets.search(text).group(1)
        return _eval(text)
    except Exception as e:
        print(e, "in", text)
        return []


if __name__ == "__main__":
    import os
    dfs = [pd.read_csv("./raw/" + f) for f in os.listdir("./raw/") if f.endswith(".csv")]
    df = pd.concat(dfs)

    df.concepts = df.concepts.progress_apply(process_work)
    df.to_csv("concepts.csv", index=False)