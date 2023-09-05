import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import json

tqdm.pandas()

df = pd.read_csv("data/concepts-v2.csv")
df["llama_concepts"] = df["llama_concepts"].apply(literal_eval)

all_concepts = set(el for l in df["llama_concepts"].tolist() for el in l)

english_words = json.loads(open("data/words_dictionary.json").read())


def is_word(word):
    """
    Check if the word is a real word
    """
    return word in all_concepts


def fix_plural(concept):
    """
    Fix plurals in the concepts
    """

    words = concept.split(" ")

    for i, word in enumerate(words):
        stem = word[:-1]

        if word.endswith("s") and len(word) > 4 and is_word(stem):
            # test if concept with plural removed is a real word
            words[i] = stem
            if " ".join(words) in all_concepts:
                continue

            words[i] = stem + "s"  # revert change

    new_concept = " ".join(words)

    return new_concept


plural_fixes = {}

for c in tqdm(all_concepts):
    new = fix_plural(c)

    if new != c:
        plural_fixes[c] = new

df["llama_concepts"] = df["llama_concepts"].progress_apply(
    lambda x: sorted(set(plural_fixes.get(c, c) for c in x))
)

df.to_csv("data/concepts-v2-fixed.csv", index=False)
