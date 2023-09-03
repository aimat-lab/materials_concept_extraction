import re
import os
import csv
from collections import Counter

STEP_SIZE = 2000

regex = re.compile(r"(\d+)_(\d+)")
inf_dir = "data/ft-xxl-backup/"


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_line_count(path):
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        return sum(1 for _ in csv_reader)


data = []

for f in os.listdir(inf_dir):
    plain_name = get_filename(inf_dir + f)
    line_count = get_line_count(inf_dir + f)
    mo = regex.search(plain_name)
    start = int(mo.group(1))
    end = int(mo.group(2))
    data.append(dict(start=start, end=end, processed=line_count, file=inf_dir + f))

end_counts = Counter(x["end"] for x in data)
to_merge = [x for x in end_counts if end_counts[x] > 1]

# group by end
grouped = {}
for x in data:
    if x["end"] not in to_merge:
        continue

    if x["end"] not in grouped:
        grouped[x["end"]] = []
    grouped[x["end"]].append(x)

# sort by start
for k in grouped:
    grouped[k] = sorted(grouped[k], key=lambda x: x["start"])

# merge csv with pandas
import pandas as pd

for end in grouped.keys():
    print(f"Merging {grouped[end]}")

    # df = pd.concat([pd.read_csv(x["file"]) for x in grouped[end]])
    # df.to_csv(inf_dir + f"{grouped[end][0]['start']:06d}_{end:06d}.csv", index=False)

# remove files which start is not % STEP_SIZE == 0
for f in data:
    if f["start"] % STEP_SIZE != 0:
        # os.remove(f["file"])
        print(f"Removing {f['file']}")
