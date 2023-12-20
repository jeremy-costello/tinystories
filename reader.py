import math
import torch
import numpy as np
from io import BytesIO
import pyarrow.parquet as pq
import pyspark.sql.types as T
import pyspark.sql.functions as F
from torch.utils.data import Dataset

from parameters import get_param_dict


def main():
    session_name = "reader-test"
    data_splits = ["train", "valid"]
    
    param_dict = get_param_dict(session_name, load_spark=True)

    for data_split in data_splits:
        shuffle_data(
            spark=param_dict["spark"],
            data_split=data_split,
            tokenized_parquet_root=param_dict["tokenizer"]["tokenized_parquet_root"],
            context_length=param_dict["model"]["context_length"],
            dataset_url=param_dict["dataset"]["dataset_url"],
            num_workers=param_dict["dataset"]["num_workers"]
        )
    
    param_dict["spark"].stop()


class ParquetDataset(Dataset):
    def __init__(self, parquet_file):
        self.parquet_table = pq.read_table(parquet_file)
        self.num_rows = len(self.parquet_table)
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        input_ids = self.parquet_table.slice(idx, 1).to_pandas()
        input_ids = input_ids["input_ids"][0]
        input_ids = np.load(BytesIO(input_ids))
        input_ids = torch.from_numpy(input_ids.astype(np.int64))
        return {
            "input_ids": input_ids
        }


def shuffle_data(spark, data_split, tokenized_parquet_root, context_length,
                 dataset_url, num_workers, partitions=None, row_limit=None):
    
    df = spark.read.parquet(f"{tokenized_parquet_root}-{data_split}.parquet")

    def split_to_ctx(arrays):
        new_binaries = []

        current_array = []
        for array in arrays:
            current_array.extend(array)
            while len(current_array) >= context_length:
                new_ndarray = np.array(current_array[:context_length])
                with BytesIO() as memfile:
                    np.save(memfile, new_ndarray)
                    new_binary = bytearray(memfile.getvalue())
                new_binaries.append(new_binary)
                current_array = current_array[context_length:]

        return new_binaries

    stc_schema = T.ArrayType(T.BinaryType())
    split_to_ctx_spark = spark.udf.register("split_to_ctx", split_to_ctx, stc_schema)

    df = df.select(F.col("input_ids"))

    df = df.orderBy(F.rand())
    df = df.withColumn("index", F.monotonically_increasing_id())

    df = df.withColumn("group", F.col("index") % math.ceil(df.count() / num_workers))
    df = df.drop(F.col("index"))

    df = df.groupBy("group").agg(F.collect_list("input_ids").alias("input_ids"))
    df = df.drop(F.col("group"))

    df = df.withColumn("input_ids", split_to_ctx_spark(F.col("input_ids")))
    df = df.withColumn("input_ids", F.explode(df["input_ids"]))

    if row_limit is not None:
        df = df.limit(row_limit)
    
    if partitions is not None:
        df.repartition(partitions)
    
    dataset_url_data_split = f"{dataset_url}/{data_split}"
    df.write.parquet(dataset_url_data_split, mode="overwrite")
    

if __name__ == "__main__":
    main()
