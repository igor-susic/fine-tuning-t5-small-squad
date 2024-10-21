#!/usr/bin/env python

import os
import torch
import gc
import json
import transformers
import torch.multiprocessing as mp

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor

class SQuADDataset:
    """
    Represents SQuAD dataset
    """
    def __init__(self, ids, questions, contexts, answers, tokenizer) -> None:
        self.ids = ids
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        
    def tokenize_input(self) -> dict:
        """
        Tokenizer will add EOS tokens at the end of the sentence and in between so it will look like:
        question: .... </s> context: .... </s>
        
        Also anything shorter will be padded with <pad> token
        """
        return self.tokenizer(self.questions, self.contexts, padding='max_length', truncation='only_second', max_length=512, return_tensors="pt")
    
    def tokenize_target(self) -> dict:
        return self.tokenizer(self.answers, padding='max_length', truncation=True, max_length=24, return_tensors="pt")
    
def load_dataset_and_preprocess(path: str, tokenizer) -> SQuADDataset:
    """
    Will load each example into the SQuAD dataset DS
    """

    fp = open(path)
    dataset: dict = json.load(fp)
    fp.close()
    
    print(f"Loading dataset SQuAD version: {dataset['version']}")
    
    data: list = dataset.get("data")
    
    ids = []
    questions = []
    contexts = []
    answers = []
    
    for chunk in data:
        print(f"Loading paragraph: {chunk.get('title')}")
        
        paragraphs: list[dict] = chunk.get("paragraphs")
        
        for paragraph in paragraphs:
            qas = paragraph.get("qas")
            context = paragraph.get("context")
            
            for example in qas:
                question = example.get("question")
                q_id = example.get("id")
                
                # We will take only one answer example
                list_of_answers = example.get("answers")
                answer = list_of_answers[0].get("text") if len(list_of_answers) > 0 else ""
                
                q = f"question: {question}"
                ctx = f"context: {context}"
                
                ids.append(q_id)
                questions.append(q)
                contexts.append(ctx)
                answers.append(answer)
                
    return SQuADDataset(ids, questions, contexts, answers, tokenizer)

class Trainer:
    def __init__(
            self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            train_data: DataLoader,
            gpu_id: int,
            save_every: int,
        ) -> None:
        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.train_data = train_data
        self.save_every = save_every

        model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(
            self, 
            input_ids: torch.Tensor, 
            input_att_mask: torch.Tensor, 
            target_ids: torch.Tensor, 
            target_att_mask: torch.Tensor
        ) -> None:

        self.optimizer.zero_grad()

        loss = self.model(
            input_ids=input_ids.to(self.gpu_id),
            attention_mask=input_att_mask.to(self.gpu_id),
            decoder_attention_mask=target_att_mask.to(self.gpu_id),
            labels=target_ids.to(self.gpu_id)
        ).loss

        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int) -> None:
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        
        self.train_data.sampler.set_epoch(epoch)

        for data in self.train_data:
            input_ids: torch.Tensor
            input_att_mask: torch.Tensor
            target_ids: torch.Tensor
            target_att_mask: torch.Tensor

            input_ids, input_att_mask, target_ids, target_att_mask = data

            self._run_batch(
                input_ids, input_att_mask, target_ids, target_att_mask
            )

    def _save_checkpoint(self, epoch: int) -> None:
        ckp = self.model.module.state_dict()
        PATH = "/home/igor_susic_admin_exponea_com/distributed_ckp.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int) -> None:
        self.model.train(True)

        for epoch in range(max_epochs):
            self._run_epoch(epoch=epoch)

        if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_checkpoint(epoch)
        
        self.model.train(False)

def setup_distributed_data_processing(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int,  world_size: int, save_every: int, total_epochs: int, batch_size: int) -> None:
    pretrained_model_used = "google-t5/t5-small"
    # We want to map immidiatley the model onto the GPU
    setup_distributed_data_processing(rank=rank, world_size=world_size)
    
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_used)
    ds: SQuADDataset = load_dataset_and_preprocess("./train-v2.0.json", tokenizer)
    tokenized_input = ds.tokenize_input()
    tokenized_target = ds.tokenize_target()

    train_data = TensorDataset(
        tokenized_input.get("input_ids"), 
        tokenized_input.get("attention_mask"), 
        tokenized_target.get("input_ids"), 
        tokenized_target.get("attention_mask")
    ) # Order of these parameters is same later once we read it

    train_sampler = DistributedSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_used, torch_dtype="auto")
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        train_data=train_dataloader,
        gpu_id=rank,
        save_every=save_every,
    )

    trainer.train(max_epochs=total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 16)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

    T5ForConditionalGeneration.load_state_dict()

