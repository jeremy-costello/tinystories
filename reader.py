import math
import pyspark.sql.types as T
import pyspark.sql.functions as F
from petastorm.etl.dataset_metadata import materialize_dataset

from parameters import get_param_dict


def main():
    session_name = "reader-test"
    data_splits = ["train", "valid"]
    
    generate_dataset(session_name, data_splits)


def generate_dataset(session_name, data_splits):
    for data_split in data_splits:
        assert data_split in ["train", "valid"]
    
    param_dict = get_param_dict(session_name, load_spark=True, load_schema=True)

    for data_split in data_splits:
        shuffle_dataset(spark=param_dict["spark"],
                        data_split=data_split,
                        tokenized_parquet_root=param_dict["tokenizer"]["tokenized_parquet_root"],
                        context_length=param_dict["model"]["context_length"],
                        dataset_url=param_dict["dataset"]["dataset_url"],
                        hdfs_home=param_dict["dataset"]["hdfs_home"],
                        row_group_size_mb=param_dict["dataset"]["row_group_size_mb"],
                        num_spark_workers=param_dict["dataset"]["num_spark_workers"],
                        petastorm_schema=param_dict["dataset"]["petastorm_schema"])
    
    param_dict["spark"].stop()


def shuffle_dataset(spark, data_split, tokenized_parquet_root, context_length, dataset_url, hdfs_home,
                    row_group_size_mb, num_spark_workers, petastorm_schema, partitions=None, row_limit=None):
    
    hadoop_url = f"{hdfs_home}/{tokenized_parquet_root}-{data_split}.parquet"

    df = spark.read.parquet(f"{tokenized_parquet_root}-{data_split}.parquet")

    def split_to_ctx(arrays):
        new_arrays = []

        current_array = []
        for array in arrays:
            current_array.extend(array)
            while len(current_array) >= context_length:
                new_arrays.append(current_array[:context_length])
                current_array = current_array[context_length:]

        return new_arrays

    udf_schema = T.ArrayType(T.ArrayType(T.LongType()))
    split_to_ctx_spark = spark.udf.register("split_to_ctx", split_to_ctx, udf_schema)

    df = df.select(F.col("input_ids"))

    df = df.orderBy(F.rand())
    df = df.withColumn("index", F.monotonically_increasing_id())

    df = df.withColumn("group", F.col("index") % math.ceil(df.count() / num_spark_workers))
    df = df.drop(F.col("index"))

    df = df.groupBy("group").agg(F.collect_list("input_ids").alias("input_ids"))
    df = df.drop(F.col("group"))

    df = df.withColumn("input_ids", split_to_ctx_spark("input_ids"))
    df = df.withColumn("input_ids", F.explode(df["input_ids"]))

    if row_limit is not None:
        df = df.limit(row_limit)
    
    if partitions is not None:
        df.repartition(partitions)
    
    dataset_url_data_split = f"{dataset_url}/{data_split}"
    with materialize_dataset(spark, hadoop_url, petastorm_schema, row_group_size_mb):
        df.write.mode("overwrite").parquet(dataset_url_data_split)


if __name__ == "__main__":
    main()
