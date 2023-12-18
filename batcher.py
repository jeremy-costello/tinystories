import pyspark.sql.functions as F
import pyspark.sql.types as T
from transformers import PreTrainedTokenizerFast

from parameters import get_param_dict


def main():
    session_name = "batcher-test"

    param_dict = get_param_dict(session_name, load_spark=True)
    create_batches(
        spark=param_dict["spark"],
        data_parquet_name=param_dict["dataset"]["data_parquet_name"],
        tokenizer_save_location=param_dict["tokenizer"]["tokenizer_save_location"],
        context_length=param_dict["model"]["context_length"],
        tokenized_parquet_name=param_dict["tokenizer"]["tokenized_parquet_name"],
        eos_token=param_dict["tokenizer"]["eos_token"]
    )
    param_dict["spark"].stop()


def create_batches(spark, data_parquet_name, tokenizer_save_location,
                   context_length, tokenized_parquet_name, eos_token):
    df = spark.read.parquet(f"{data_parquet_name}.parquet")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_save_location)

    tokenizer.bos_token = eos_token
    tokenizer.eos_token = eos_token
    tokenizer.pad_token = eos_token


    def tokenize_udf(text):
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True
        )
        return list(outputs.values())


    schema = T.ArrayType(T.ArrayType(T.StringType()))
    tokenize_udf_spark = spark.udf.register("tokenize_udf", tokenize_udf, schema)

    df = df.withColumnRenamed("value", "text")

    index_name = "index"
    df = df.withColumn(index_name, F.monotonically_increasing_id())
    df = df.select([index_name] + [col_name for col_name in df.columns if col_name != index_name])

    df = df.withColumn("result", tokenize_udf_spark(F.col("text")))
    df = df.drop(F.col("text"))

    two_nest_list = ["input_ids", "token_type_ids", "attention_mask"]
    one_nest_list = ["length", "overflow_to_sample_mapping"]

    for i, column_name in enumerate(two_nest_list + one_nest_list):
        df = df.withColumn(column_name, F.col("result")[i])

    df = df.drop("result")

    df = df.withColumn("new", F.arrays_zip("input_ids", "token_type_ids", "attention_mask", "length", "overflow_to_sample_mapping")) \
            .withColumn("new", F.explode("new")) \
            .select("index",
                    F.col("new.input_ids").alias("input_ids"),
                    F.col("new.token_type_ids").alias("token_type_ids"),
                    F.col("new.attention_mask").alias("attention_mask"),
                    F.col("new.length").alias("length"),
                    F.col("new.overflow_to_sample_mapping").alias("overflow_to_sample_mapping"))

    for column_name in two_nest_list:
        df = df.withColumn(column_name, F.split(F.regexp_replace(F.col(column_name), "[\\[\\]]", ""), ", "))

    for column_name in two_nest_list:
        df = df.withColumn(column_name, df[column_name].cast("array<bigint>"))

    for column_name in one_nest_list:
        df = df.withColumn(column_name, F.col(column_name).astype(T.LongType()))
    
    df.write.parquet(f"{tokenized_parquet_name}.parquet", mode="overwrite")


if __name__ == "__main__":
    main()
