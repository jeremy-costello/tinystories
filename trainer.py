import math
import torch
import lightning as L
from pyspark.sql import SparkSession
from petastorm.reader import make_reader
from petastorm.pytorch import DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from transformers import GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast

from reader import shuffle_dataset
from parameters import get_param_dict


def main():
    session_name = "trainer-test"

    param_dict = get_param_dict(session_name, load_schema=True)
    train_model(session_name, param_dict)


def train_model(session_name, param_dict):
    n_positions = 2 * param_dict["model"]["context_length"]

    num_layers = param_dict["model"]["num_layers"]
    assert num_layers % 2 == 0
    attention_types = [[["global", "local"], num_layers // 2]]

    config = GPTNeoConfig(
        vocab_size=param_dict["tokenizer"]["vocab_size"],
        max_position_embeddings=n_positions,
        hidden_size=param_dict["model"]["hidden_size"],
        num_layers=param_dict["model"]["num_layers"],
        attention_types=attention_types,
        num_heads=param_dict["model"]["num_heads"],
        resid_dropout=param_dict["model"]["resid_pdrop"],
        embed_dropout=param_dict["model"]["embd_pdrop"],
        attention_dropout=param_dict["model"]["attn_pdrop"],
        bos_token_id=0,
        eos_token_id=0
    )

    model = Transformer(
        config=config,
        param_dict=param_dict
    )

    mlf_logger = MLFlowLogger(
        experiment_name=session_name,
        tracking_uri="file:./mlruns")
    
    trainer = L.Trainer(
        accelerator=param_dict["training"]["accelerator"],
        devices=param_dict["training"]["devices"],
        strategy=param_dict["training"]["strategy"],
        precision=param_dict["training"]["precision"],
        logger=mlf_logger,
        max_steps=param_dict["training"]["max_steps"],
        accumulate_grad_batches=param_dict["training"]["accumulate_grad_batches"],
        gradient_clip_val=param_dict["training"]["gradient_clip_val"])
    
    trainer.fit(model)


class Transformer(L.LightningModule):
    def __init__(self, config, param_dict):
        super().__init__()

        self.shuffle_data = param_dict["training"]["shuffle_data"]
        self.learning_rate = param_dict["training"]["learning_rate"]
        self.betas = param_dict["training"]["betas"]
        self.final_lr_multiplier = param_dict["training"]["final_lr_multiplier"]
        self.max_steps = param_dict["training"]["max_steps"]
        self.warmup_steps = param_dict["training"]["warmup_steps"]
        self.batch_size = param_dict["training"]["batch_size"]
        self.save_model_interval = param_dict["training"]["save_model_interval"]
        self.dataset_url = param_dict["dataset"]["dataset_url"]

        self.param_dict = param_dict

        self.model = GPTNeoForCausalLM(config)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=param_dict["tokenizer"]["tokenizer_save_location"]
        )

        self.step_loss = 0
    
    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        outputs = self(input_ids, attention_mask)

        loss = outputs["loss"]
        return loss

    def on_before_backward(self, loss):
        self.step_loss += loss
    
    def on_before_optimizer_step(self, optimizer):
        self.logger.experiment.log_metric(self.logger.run_id, "loss", self.step_loss)
        self.step_loss = 0
    
    def on_before_zero_grad(self, optimizer):
        if self.global_step % self.save_model_interval == 0:
            self.save_huggingface_model()
    
    def on_train_end(self):
        self.save_huggingface_model()
    
    def calculate_learning_rate(self, step):
        real_step = step + 1
        if real_step <= self.warmup_steps:
            lr_mult = real_step / self.warmup_steps
        else:
            lr_mult = self.final_lr_multiplier  + 0.5 * (1 - self.final_lr_multiplier) * \
                (1 + math.cos(math.pi * (real_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
        
        self.logger.experiment.log_metric(self.logger.run_id, "lr", self.learning_rate * lr_mult, step=self.global_step)
        return lr_mult

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      betas=(self.betas))

        lr_lambda = lambda step: self.calculate_learning_rate(step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
    
    def on_epoch_start(self):
        self.trainer.train_dataloader = self.train_dataloader()
    
    def on_epoch_end(self):
        self.trainer.train_dataloader.__exit__(None, None, None)
    
    def train_dataloader(self):
        session_name = "petastorm-reader"

        if self.shuffle_data:
            spark = SparkSession.builder.appName(session_name).getOrCreate()
            shuffle_dataset(spark=spark,
                            tokenized_parquet_name=self.param_dict["tokenizer"]["tokenized_parquet_name"],
                            context_length=self.param_dict["model"]["context_length"],
                            dataset_url=self.param_dict["dataset"]["dataset_url"],
                            hdfs_home=self.param_dict["dataset"]["hdfs_home"],
                            row_group_size_mb=self.param_dict["dataset"]["row_group_size_mb"],
                            num_spark_workers=self.param_dict["dataset"]["num_spark_workers"],
                            petastorm_schema=self.param_dict["dataset"]["petastorm_schema"])
            spark.stop()

        petastorm_reader = make_reader(self.dataset_url)
        loader = DataLoader(petastorm_reader, batch_size=self.batch_size)
        return loader
    
    def save_huggingface_model(self):
        self.model.save_pretrained(f"./models/{self.logger.run_id}/{self.global_step}")


if __name__ == "__main__":
    main()
