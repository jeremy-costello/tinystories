import os
import glob
import json
import random
from tqdm import tqdm


ENDING = " \n<|endoftext|>\n"
TRAIN_RATIO = 0.99
# GPT-3.5 or GPT-4
ACCEPTED_SOURCES = ["GPT-4"]


train_file = "./texts/tinystories_tar-train.txt"
valid_file = "./texts/tinystories_tar-valid.txt"
if os.path.isfile(train_file):
    os.remove(train_file)
if os.path.isfile(valid_file):
    os.remove(valid_file)

duplicate_count = 0
story_set = set()

json_files = glob.glob(f"data/*.json")
with open(train_file, "a") as write_train_text:
    with open(valid_file, "a") as write_valid_text:
        for json_file in tqdm(json_files):
            with open(json_file, "r") as read_json:
                data = json.load(read_json)
            for datum in tqdm(data, leave=False):
                if datum["source"] in ACCEPTED_SOURCES:
                    story = datum["story"].strip()
                    if story not in story_set:
                        story_set.add(story)
                        if random.random() >= TRAIN_RATIO:
                            write_valid_text.write(story + ENDING)
                        else:
                            write_train_text.write(story + ENDING)
                    else:
                        duplicate_count += 1

print(duplicate_count)
