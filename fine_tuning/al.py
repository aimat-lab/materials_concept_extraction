from fire import Fire
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


datapoint = "({i}) [A] {abstract} [/A]\n[K] {keywords} [/K]\n"


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
        max_tokens=100,
    )

    return r["choices"][0]["message"]["content"]


def main(
    input_file="",
    batch_size=5,
    start_iteration=2,
    end_iterations=4,
):
    data = pd.read_csv(input_file)

    for index in range(start_iteration * batch_size, end_iterations * batch_size, batch_size):
        batch = data.iloc[index : index + batch_size]

        batch_datapoints = [
            datapoint.format(
                i=i + 1,
                abstract=abstract,
                keywords=keywords,
            )
            for i, (abstract, keywords) in enumerate(
                zip(batch.abstract, batch.concepts)
            )
        ]

        batch_datapoints = "\n".join(batch_datapoints)

        sys_prompt = format_prompt("fine_tuning/system_prompt.txt", k=batch_size)
        user_prompt = format_prompt("fine_tuning/prompt.txt", data=batch_datapoints)

        print(
            sys_prompt,
            user_prompt
        )

        response = perform_request(
            sys_prompt,
            user_prompt,
        )

        print("===>", response)


if __name__ == "__main__":
    Fire(main)
