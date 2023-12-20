import os


param_dict = {
    "hadoop_install": "~/hadoop/hadoop-3.3.6",
    "tokenizer": {
        "eos_token": "<|endoftext|>",
        "tokenizer_save_location": "./tokenizers/tinystories.json",
        "vocab_size": 10000,
        "batch_size": 256,
        "tokenized_parquet_root": "tinystories-tokenized"
    },
    "dataset": {
        "text_folder": "./texts",
        "raw_text_file_root": "TinyStoriesV2-GPT4",
        "preprocessed_file_root": "tinystories",
        "splitter": "<|SPLIT|>\n",
        "data_parquet_root": "tinystories",
        "dataset_url": "./petastorm_data",
        "num_workers": 3
    },
    "training": {
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "deepspeed_stage_1",
        "precision": "bf16-mixed",
        "float32_matmul_precision": "medium",
        "reload_dataloaders": False,
        "shuffle_data": False,
        "max_steps": 20000,
        "warmup_steps": 1000,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 1.0,
        "val_check_interval": 640,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
        "final_lr_multiplier": 0.1
    },
    "model": {
        "context_length": 512,
        "hidden_size": 64,
        "num_layers": 8,
        "num_heads": 16,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1
    }
}


def get_param_dict(session_name, load_spark=False):
    if load_spark:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName(session_name).getOrCreate()
        param_dict["spark"] = spark
    
    dataset_url = param_dict["dataset"]["dataset_url"].lstrip(".").strip("/")
    full_dataset_url = f"file://{os.getcwd()}/{dataset_url}"
    param_dict["dataset"]["dataset_url"] = full_dataset_url

    return param_dict
