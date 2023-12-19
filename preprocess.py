from parameters import get_param_dict


def main():
    session_name = "preprocess-test"

    param_dict = get_param_dict(session_name)
    preprocess_for_spark(
        text_folder=param_dict["dataset"]["text_folder"],
        raw_text_file_root=param_dict["dataset"]["raw_text_file_root"],
        preprocessed_file_root=param_dict["dataset"]["preprocessed_file_root"],
        eos_token=param_dict["tokenizer"]["eos_token"],
        splitter=param_dict["dataset"]["splitter"]
    )


def preprocess_for_spark(text_folder, raw_text_file_root, preprocessed_file_root, eos_token, splitter, max_stories=None):
    """
    DOCSTRING
    """

    for data_split in ["train", "valid"]:
        stories = []

        lines = []
        num_stories = 0

        raw_text_file = f"{text_folder}/{raw_text_file_root}-{data_split}.txt"
        with open(raw_text_file, "r") as f:
            for line in f.readlines():
                stripped = line.strip()
                if stripped == eos_token:
                    story = "\n".join(lines) + "\n" + eos_token + splitter
                    stories.append(story)
                    lines = []
                    num_stories += 1
                    if num_stories == max_stories:
                        break
                elif stripped:
                    lines.append(stripped)
        
        preprocessed_file = f"{text_folder}/{preprocessed_file_root}-{data_split}.txt"
        with open(preprocessed_file, "w") as f:
            for story in stories:
                f.write(story)


if __name__ == "__main__":
    main()
