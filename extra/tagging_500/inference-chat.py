from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from peft import PeftModel
import pandas as pd
import time
import fire
import os


def load_prompt(prompt_file):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    return prompt


def get_format_prompt(prompt_file):
    prompt = load_prompt(prompt_file)

    def format_prompt(**kwargs):
        return prompt.format(**kwargs)

    return format_prompt


def main(
    input_file="./data/inference.csv",
    llama_variant="7B",
    model_id="finetuned",
    use_base_model=False,
    batch_size=5,
    inf_limit=5,
    max_new_tokens=512,
    prompt_file="data/prompt.txt",
):
    MODEL_PATH = f"./models/{llama_variant}/{model_id}"

    tokenizer = LlamaTokenizerFast.from_pretrained(
        f"./llama-{llama_variant}/", return_tensors="pt", paddding_side="left"
    )
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        f"./llama-{llama_variant}/", load_in_8bit=True, device_map="auto"
    )
    if not use_base_model:
        model = PeftModel.from_pretrained(model, MODEL_PATH)

    model.eval()

    format_prompt = get_format_prompt(prompt_file)

    df = pd.read_csv(input_file).head(inf_limit)
    df.abstract = df.abstract.apply(
        lambda abstract: "<s>" + format_prompt(ABSTRACT=abstract)
    )  # prepare

    abstracts = list(df.abstract)
    df["generated"] = None

    start_time = time.time()
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i : i + batch_size]

        # Pad the sequences in the batch to the same length
        inputs = tokenizer(
            batch,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0,
        )

        generated_ids = [output.detach().cpu().numpy() for output in outputs]
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        df.loc[i : i + batch_size - 1, "generated"] = generated_texts

    print("Inference took: ", time.time() - start_time, "\n")

    settings = {
        "llama_variant": llama_variant,
        "model_id": model_id,
        "use_base_model": use_base_model,
        "batch_size": batch_size,
        "inf_limit": inf_limit,
        "max_new_tokens": max_new_tokens,
    }
    print("Settings:", settings)

    df = df[["id", "generated"]]
    df.generated = df.generated.apply(
        lambda t: t.split("Assistant: ")[1].strip()
    )  # postprocess, only keep generated text
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    OUTPUT_PATH = f"./data/inference_{llama_variant}/{model_id}/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df.to_csv(f"{OUTPUT_PATH}{current_time}_{model_id}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
