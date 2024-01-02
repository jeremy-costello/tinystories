# tinystories
Recreating the training loop for [TinyStories](https://arxiv.org/abs/2305.07759).

# howto
This requires Hadoop. I used [this guide](https://kontext.tech/article/978/install-hadoop-332-in-wsl-on-windows) to install it locally on WSL.

1. Enter parameters into ```parameters.py```
2. Put the TinyStories [train](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-train.txt) and [validation](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-valid.txt) files into ```./texts```
3. Run Hadoop: ```sbin/start-dfs.sh```
4. Run ```preprocess.py``` (removes empty lines, adds custom line split separator for Spark)
5. Put the preprocessed text files onto the Hadoop FS: ```hadoop fs -put tinystories-train.txt``` and ```hadoop fs -put tinystories-valid.txt```
6. Run ```tokenizer.py``` (trains a tokenizer with Huggingface)
7. Run ```batcher.py``` (tokenizes each story and saves to a parquet file)
8. Run ```trainer.py``` (trains a GPT2 model with PyTorch Lightning & MLFlow tracking)

# todo
- requirements.txt / poetry installer
- checkpoint saving and restarting
- add docstrings
- add explanations for parameters
- find good settings for stable training
- use hydra for config
- use torchmetrics for loss
- use flashattention2 model and cross-entropy loss
- use tensorboard instead of MLflow
- use deepspeed optimizer/scheduler
