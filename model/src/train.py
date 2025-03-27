# model/train.py
# 모델 학습
import os
import json

import torch
from torch.utils.data import RandomSampler, DataLoader
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from model.src.processor import Processor, load_and_cache_examples
from common.utils import (
    CONFIG_CLASSES,
    TOKENIZER_CLASS,
    MODEL_FOR_TOKEN_CLASSIFICATION
)
from model.src.evaluation import Evaluator

class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self):
        processor = Processor(self.args)
        # 1. 데이터 토크나이징까지 해서 모델 넣기 전까지 상태로 전처리
        # 2. 학습 루프 생성
        # 3. 평가
        # 4. 저장
        
        tokenizer = TOKENIZER_CLASS[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case
        )
        config = CONFIG_CLASSES[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            num_labels=len(processor.categories),
            id2label={str(i): label for i, label in enumerate(processor.categories)},
            label2id={label: i for i, label in enumerate(processor.categories)},
        )
        model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        self.args.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        model.to(self.args.device)

        # Data Preprocessing
        train_dataset = load_and_cache_examples(self.args, tokenizer, mode="train") if self.args.train_file else None
        dev_dataset = load_and_cache_examples(self.args, tokenizer, mode="dev") if self.args.dev_file else None
        test_dataset = load_and_cache_examples(self.args, tokenizer, mode="test") if self.args.test_file else None

        if dev_dataset == None:
            self.args.evaluate_test_during_training = True
        
        global_step, tr_loss = self._train(model, train_dataset, dev_dataset, test_dataset)
        print(f" global_step = {global_step}, average loss = {tr_loss}")

    def _train(self, 
               model, 
               train_dataset, 
               dev_dataset = None, 
               test_dataset = None):
        """
        Train Loop
        Important Hyperparameters:
        - batch_size
        - num_train_epochs
        - max_steps
        - gradient_accumulation_steps
        """
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, 
                                      sampler=train_sampler,
                                      batch_size=self.args.train_batch_size)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # Prepare Optimizer and Scheduler(linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)

        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))
        
        # Train!
        global_step = 0
        tr_loss = 0.0

        model.zero_grad()
        mb = master_bar(range(int(self.args.num_train_epochs)))
        for epoch in mb:
            epoch_iterator = progress_bar(train_dataloader, parent=mb)
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        if self.args.evaluate_test_during_training:
                            Evaluator(self.args)._evaluate(model, test_dataset, "test", global_step)
                        else:
                            Evaluator(self.args)._evaluate(model, dev_dataset, "dev", global_step)
                
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)

                        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                        print(f"Saving model checkpoint to {output_dir}")

                        if self.args.save_optimizer:
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if 0 < self.args.max_steps > 0 and global_step > self.args.max_steps:
                    break
            
            mb.write(f"Epoch {epoch+1} done.")

            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                break

        return global_step, tr_loss/global_step