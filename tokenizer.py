from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from parameters import get_param_dict


def main():
    session_name = "tokenizer-test"

    param_dict = get_param_dict(session_name, load_spark=True)
    train_tokenizer(
        spark=param_dict["spark"],
        preprocessed_file=param_dict["dataset"]["preprocessed_file"],
        eos_token=param_dict["tokenizer"]["eos_token"],
        splitter=param_dict["dataset"]["splitter"],
        data_parquet_name=param_dict["dataset"]["data_parquet_name"],
        tokenizer_save_location=param_dict["tokenizer"]["tokenizer_save_location"],
        vocab_size=param_dict["tokenizer"]["vocab_size"],
        tokenizer_batch_size=param_dict["tokenizer"]["batch_size"]
    )
    param_dict["spark"].stop()


def train_tokenizer(spark, preprocessed_file, eos_token, splitter, data_parquet_name,
                    tokenizer_save_location, vocab_size, tokenizer_batch_size):
    tok_class = PySparkTokenizer(
        spark=spark,
        preprocessed_file=preprocessed_file,
        splitter=splitter,
        data_parquet_name=data_parquet_name
    )
    tok_class.train_tokenizer(
        vocab_size=vocab_size,
        special_tokens=[eos_token],
        batch_size=tokenizer_batch_size
    )
    tok_class.save_tokenizer(tokenizer_save_location)


class PySparkTokenizer:
    def __init__(self, spark, preprocessed_file, splitter, data_parquet_name, load_from_parquet=False):
        if load_from_parquet:
            assert data_parquet_name is not None
            self.df = spark.read.parquet(f"{data_parquet_name}.parquet")
        else:
            self.df = spark.read.text(preprocessed_file, lineSep=splitter)
            self.df.write.parquet(f"{data_parquet_name}.parquet", mode="overwrite")
    
    def batch_iterator(self, batch_size):
        iterator = self.df.rdd.toLocalIterator()

        batch = []
        for i, row in enumerate(iterator):
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = []
            batch.append(row.value)
        if batch:
            yield batch

    def train_tokenizer(self, vocab_size, special_tokens, batch_size):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens
        )

        length = self.df.count()
        tokenizer.train_from_iterator(self.batch_iterator(batch_size), trainer=trainer, length=length)

        self.tokenizer = tokenizer
    
    def save_tokenizer(self, save_location):
        self.tokenizer.save(save_location)


if __name__ == "__main__":
    main()
