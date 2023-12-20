import math
import torch
import lightning as L
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers.pytorch_utils import Conv1D

from reader import shuffle_parquet_data, ParquetDataset
from parameters import get_param_dict


def main():
    session_name = "trainer-test"

    param_dict = get_param_dict(session_name, load_spark=False)
    train_model(session_name, param_dict)


def train_model(session_name, param_dict):
    if param_dict["training"]["float32_matmul_precision"] is not None:
        torch.set_float32_matmul_precision(
            param_dict["training"]["float32_matmul_precision"]
        )
    
    n_positions = param_dict["model"]["context_length"]

    config = GPT2Config(
        vocab_size=param_dict["tokenizer"]["vocab_size"],
        n_positions=n_positions,
        n_embd=param_dict["model"]["hidden_size"],
        n_layer=param_dict["model"]["num_layers"],
        n_head=param_dict["model"]["num_heads"],
        resid_pdrop=param_dict["model"]["resid_pdrop"],
        embd_pdrop=param_dict["model"]["embd_pdrop"],
        attn_pdrop=param_dict["model"]["attn_pdrop"],
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
        gradient_clip_val=param_dict["training"]["gradient_clip_val"],
        val_check_interval=param_dict["training"]["val_check_interval"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0)
    
    reload_dataloaders = param_dict["training"]["reload_dataloaders"]
    if reload_dataloaders:
        shuffle_parquet_data(
            spark=param_dict["spark"],
            data_split="valid",
            tokenized_parquet_root=param_dict["tokenizer"]["tokenized_parquet_root"],
            context_length=param_dict["model"]["context_length"],
            dataset_url=param_dict["dataset"]["dataset_url"],
            num_workers=param_dict["dataset"]["num_workers"]
        )
    # param_dict["spark"].stop()

    dataset_url = param_dict["dataset"]["dataset_url"]
    dataset_url_data_split = f"{dataset_url}/valid"
    valid_dataset = ParquetDataset(dataset_url_data_split)
    
    num_workers = param_dict["dataset"]["num_workers"]
    valid_loader = DataLoader(valid_dataset,
                              batch_size=param_dict["training"]["batch_size"],
                              shuffle=False,
                              pin_memory=True,
                              num_workers=num_workers)
    trainer.fit(model, val_dataloaders=[valid_loader])


class Transformer(L.LightningModule):
    def __init__(self, config, param_dict):
        super().__init__()

        self.reload_dataloaders = param_dict["training"]["reload_dataloaders"]
        self.shuffle_data = param_dict["training"]["shuffle_data"]
        self.learning_rate = param_dict["training"]["learning_rate"]
        self.betas = param_dict["training"]["betas"]
        self.weight_decay = param_dict["training"]["weight_decay"]
        self.final_lr_multiplier = param_dict["training"]["final_lr_multiplier"]
        self.max_steps = param_dict["training"]["max_steps"]
        self.warmup_steps = param_dict["training"]["warmup_steps"]
        self.batch_size = param_dict["training"]["batch_size"]
        self.dataset_url = param_dict["dataset"]["dataset_url"]

        self.param_dict = param_dict

        self.model = GPT2LMHeadModel(config)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=param_dict["tokenizer"]["tokenizer_save_location"]
        )
        self.tokenizer.bos_token = param_dict["tokenizer"]["eos_token"]
        self.tokenizer.eos_token = param_dict["tokenizer"]["eos_token"]

        self.logger_step = 0
        self.datasets = dict()
    
    def forward(self, input_ids):
        return self.model(
            input_ids=input_ids,
            labels=input_ids
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        outputs = self(input_ids)

        loss = outputs.loss
        self.average_train_loss += loss
        return loss

    def on_train_start(self):
        self.average_train_loss = 0
        self.save_huggingface_model()
    
    def on_before_optimizer_step(self, optimizer):
        self.logger.experiment.log_metric(self.logger.run_id, "train_loss", self.average_train_loss, step=self.logger_step)
        self.average_train_loss = 0
        self.logger_step += 1
    
    def on_train_end(self):
        self.save_huggingface_model()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        outputs = self(input_ids)

        loss = outputs.loss
        self.average_valid_loss = \
            batch_idx / (batch_idx + 1) * self.average_valid_loss + 1 / (batch_idx + 1) * loss
        
        return loss

    def on_validation_start(self):
        self.average_valid_loss = 0
    
    def on_validation_end(self):
        self.logger.experiment.log_metric(self.logger.run_id, "valid_loss", self.average_valid_loss, step=self.logger_step)
        self.save_huggingface_model()
    
    def calculate_learning_rate(self, step):
        real_step = step + 1
        if real_step <= self.warmup_steps:
            lr_mult = real_step / self.warmup_steps
        else:
            lr_mult = self.final_lr_multiplier  + 0.5 * (1 - self.final_lr_multiplier) * \
                (1 + math.cos(math.pi * (real_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
        
        self.logger.experiment.log_metric(self.logger.run_id, "lr", self.learning_rate * lr_mult, step=self.logger_step)
        return lr_mult

    def get_weight_decay_groups(self):
        # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Conv1D)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        lm_head_name = "model.lm_head.weight"

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                # same parameter as 'model.transformer.wte.weight'
                if fpn == lm_head_name:
                    continue
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def configure_optimizers(self):
        optim_groups = self.get_weight_decay_groups()
        optimizer = torch.optim.AdamW(optim_groups,
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
        self.within_epoch_step = 0
    
    def on_epoch_end(self):
        self.trainer.train_dataloader.__exit__(None, None, None)
    
    def get_dataloader(self, data_split, shuffle_data):
        session_name = f"petastorm-reader-{data_split}"

        if self.reload_dataloaders and (shuffle_data or self.logger_step == 0):
            spark = SparkSession.builder.appName(session_name).getOrCreate()
            shuffle_parquet_data(
                spark=spark,
                data_split=data_split,
                tokenized_parquet_root=self.param_dict["tokenizer"]["tokenized_parquet_root"],
                context_length=self.param_dict["model"]["context_length"],
                dataset_url=self.dataset_url,
                num_workers=self.param_dict["dataset"]["num_workers"]
            )
            spark.stop()

        dataset_url_data_split = f"{self.dataset_url}/{data_split}"
        dataset = ParquetDataset(dataset_url_data_split)
        self.datasets[data_split] = dataset

        num_workers = self.param_dict["dataset"]["num_workers"]
        loader = DataLoader(self.datasets[data_split],
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=num_workers)
        return loader

    def train_dataloader(self):
        loader = self.get_dataloader("train", shuffle_data=self.shuffle_data)
        return loader
        
    def save_huggingface_model(self):
        self.model.save_pretrained(f"./models/{self.logger.run_id}/{self.logger_step}")


if __name__ == "__main__":
    main()
