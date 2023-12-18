param_dict = {
    "hadoop_install": "~/hadoop/hadoop-3.3.6",
    "tokenizer": {
        "eos_token": "<|endoftext|>",
        "tokenizer_save_location": "./tokenizers/tinystories.json",
        "vocab_size": 10000,
        "batch_size": 256,
        "tokenized_parquet_name": "tinystories-tokenized"
    },
    "dataset": {
        "raw_text_file": "./texts/TinyStoriesV2-GPT4-train.txt",
        "preprocessed_file": "tinystories.txt",
        "splitter": "<|SPLIT|>\n",
        "data_parquet_name": "tinystories",
        "dataset_url": "file:///home/jeremy/python/gan/petastorm_data",
        "hdfs_home": "hdfs://localhost:9000/user/jeremy",
        "row_group_size_mb": 128,
        "num_spark_workers": 4
    },
    "training": {
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "deepspeed_stage_1",
        "precision": "bf16-mixed",
        "shuffle_data": True,
        "max_steps": 100000,
        "warmup_steps": 20000,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "betas": (0.9, 0.95),
        "final_lr_multiplier": 0.1,
        "save_model_interval": 1000
    },
    "model": {
        "context_length": 512,
        "hidden_size": 64,
        "num_layers": 8,
        "num_heads": 16,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0
    }
}


def get_param_dict(session_name, load_spark=False, load_schema=False):
    if load_spark:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName(session_name).getOrCreate()
        param_dict["spark"] = spark

    if load_schema:
        import numpy as np
        from petastorm.codecs import NdarrayCodec
        from petastorm.unischema import Unischema, UnischemaField
        
        context_length = param_dict["model"]["context_length"]

        petastorm_schema = Unischema("schema", [
            UnischemaField("input_id", np.int64, (context_length,), NdarrayCodec(), False)
        ])

        param_dict["dataset"]["petastorm_schema"] = petastorm_schema

    return param_dict
