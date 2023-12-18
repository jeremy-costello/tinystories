# tinystories
Recreating the training loop for [TinyStories](https://arxiv.org/abs/2305.07759).

# howto
This requires Hadoop. I used [this guide](https://kontext.tech/article/978/install-hadoop-332-in-wsl-on-windows) to install it locally on WSL.

1. Enter parameters ```into parameters.py```
2. Put the [TinyStories text](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-train.txt) file into ```./texts```
3. Run Hadoop: ```sbin/start-dfs.sh```
4. Run ```preprocess.py``` (remove empty lines, add custom line split separator for Spark)
5. Put the preprocessed text file onto the Hadoop FS: ```hadoop fs -put tinystories.txt```
6. Run ```tokenizer.py``` (trains a tokenizer with Huggingface)
7. Run ```batcher.py``` (tokenizes each story and saves to a parquet file)
8. Run ```trainer.py``` (trains a GPTNeo model with PyTorch Lightning & MLFlow)

# todo
- use GPT2 instead of GPTNeo
- requirements.txt / poetry installer
- checkpoint saving and restarting
- tracking parameter, activation, gradient norms
- validation set and validation loss tracking
- add docstrings
- add explanations for parameters
