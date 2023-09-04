from fire import Fire
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


datapoint = "({i}) [A] {abstract} [/A]"


def format_prompt(prompt_file, **kwargs):
    prompt = open(prompt_file).read()
    return prompt.format(**kwargs)


def perform_request(system_prompt, prompt):
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=512,
    )

    return r["choices"][0]["message"]["content"]


def main(
    input_file="fine_tuning/0000-1000.csv",
    start=100,
    end=120,
):
    data = pd.read_csv(input_file).iloc[start:end]

    l = []

    for id, abstract in tqdm(zip(data.id, data.abstract), total=len(data)):
        sys_prompt = format_prompt("fine_tuning/system_prompt.txt")
        user_prompt = format_prompt(
            "fine_tuning/prompt.txt", data=abstract, k=len(abstract)
        )

        response = perform_request(
            sys_prompt,
            user_prompt,
        )

        l.append({"id": id, "openai": response})

    df = pd.DataFrame(l)
    df.to_csv(f"fine_tuning/openai_{start}-{end}.csv", index=False)


if __name__ == "__main__":
    Fire(main)
