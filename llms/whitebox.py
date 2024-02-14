import torch
from torch.utils.data import DataLoader
from accelerate.state import PartialState
from accelerate.utils import release_memory, InitProcessGroupKwargs
import datasets 
from datasets import Dataset
datasets.disable_progress_bar()
from datetime import timedelta
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["WANDB_LOG_MODEL"]="false"

from tqdm.auto import tqdm

from utils.util import get_answer_start_idx
from utils.loggers import loggers

from transformers import (
    AdamW,
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    DataCollatorWithPadding,
    get_constant_schedule_with_warmup,
)

from accelerate import Accelerator

torch.cuda.empty_cache()
torch.set_printoptions(threshold=10_000)

class Whitebox_LLM():
    '''
    This class implements the Whitebox_LLM model, such as Llama.
    '''
    def __init__(
            self, 
            config,
        ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                config['critic_model'],
                truncation_side='left',
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=96000))
        self.accelerator = Accelerator(
                split_batches=False,
                mixed_precision='fp16',
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                log_with='wandb' if self.config.get("log_with_wandb", False) else None,
                project_dir='logs' if self.config.get("log_with_wandb", False) else None,
                device_placement=True,
                kwargs_handlers=[kwargs]
            )
        
        self.mode = config["critic_mode"] 
        if self.mode == 'generation':
            self.model = AutoModelForCausalLM.from_pretrained(
                config['critic_model'],
                trust_remote_code=True,
            )

        elif self.mode == 'classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config['critic_model'],
                trust_remote_code=True,
                num_labels=1, 
            )
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
        else:
            raise NotImplementedError
        
        self.model.config.use_cache = False if "phi" in self.config["critic_model"].lower() else True
        self.model.config.pretraining_tp = 1
        
        if self.tokenizer.pad_token is None:
            self.accelerator.print("Adding pad token to the tokenizer...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.answer_token = self.tokenizer.encode("\nA: ", return_tensors="pt", add_special_tokens=False)[0, 1:]

        self.optimizer = AdamW(
                self.model.parameters(), 
                lr=config['learning_rate'] * self.accelerator.gradient_accumulation_steps,
                weight_decay=0.01,
            )
        self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=config["warmup_steps"], 
        )
        
        self.accelerator.print(
            f"Distributed: {self.accelerator.distributed_type}, Mixed precision: {self.accelerator.mixed_precision}"
        )

    @PartialState().on_main_process
    def build_dataset(self, positive_texts, negative_texts, save_to):    
        pos_len, neg_len = len(positive_texts), len(negative_texts)
        labels = - torch.ones(pos_len + neg_len)
        labels[:pos_len] = 1. 
        
        input_texts = positive_texts + negative_texts   
        temp_dataset = Dataset.from_dict({
                "texts": input_texts,
                "labels": labels
            }).with_format("torch")
        batch_dataset = temp_dataset.map(
            lambda x: self.tokenizer(
                    x["texts"], 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True, 
                    add_special_tokens=self.config["add_special_tokens"],
                ),
            remove_columns=["texts"],
            batched=True)
        
        batch_dataset.save_to_disk(save_to)
        print(f"\nDataset saved to {save_to}\n")

        tr_text = f"\npos_len: {pos_len}\nneg_len: {neg_len}\n\n" +\
        f"\n\n{'-'*20}\n\n".join([t + f" (Label: {l})" for t, l in zip(input_texts, labels.tolist())])
        loggers["train"].info(f"\n{'='*20}\n{tr_text}\n\n")


    def build_dataloader(self, batch_dataset):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader_params = {
            "batch_size": self.config["batch_size"],
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
            "shuffle": True,
        }
        batch_dataloader = self.accelerator.prepare(DataLoader(batch_dataset, **dataloader_params))
        return batch_dataloader
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        labels = inputs.pop("labels").type(torch.LongTensor).to(self.accelerator.device)
        
        inputs = inputs.to(self.accelerator.device)
        outputs = model(**inputs)
        output_logits = outputs.get("logits") # (B, seq_len, V) / (B, 1)
                
        input_ids = inputs["input_ids"].detach() # (B, seq_len)
        attention_mask = inputs["attention_mask"].detach() 
        
        alpha = self.config["l2_reg_coef"]
        energy_temp = self.config["energy_temp"]
        l2_loss = 0.
        
        if self.mode == 'generation':
            logits = output_logits.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1) # (B, seq_len) 
            
            for i in range(input_ids.shape[0]):
                answer_start_from = get_answer_start_idx(input_ids[i], self.answer_token)
                attention_mask[i, :answer_start_from] = 0.
            
            energies = - (logits * attention_mask).mean(axis=-1) # (B)
                        
        if self.mode == 'classification':
            energies = - output_logits.squeeze(-1) # (B)
            
        pos_energy = energies[labels > 0] / energy_temp
        neg_energy = energies[labels < 0] / energy_temp
        
        # when batch size is really small, pos_energy and neg_energy can be empty
        # which will cause nan loss
        if pos_energy.shape[0] == 0:
            pos_energy = torch.zeros(1).to(self.accelerator.device)
        if neg_energy.shape[0] == 0:
            neg_energy = torch.zeros(1).to(self.accelerator.device)
        
        ml_loss = pos_energy.mean() - neg_energy.mean()
        
        if alpha != 0:
            l2_loss = alpha * energies.square().mean()
            
        loss = ml_loss + l2_loss
                
        self.accelerator.log({"total_loss": loss.item()})
        self.accelerator.log({"l2_loss": l2_loss.item() if alpha > 0 else 0.})
        self.accelerator.log({"ml_loss": ml_loss.item()})
        self.accelerator.log({"pos_energy": pos_energy.mean().item()})
        self.accelerator.log({"neg_energy": neg_energy.mean().item()})        
        
        loggers["tensor"].info(
            f"\n{'='*20}\n\ninput_ids:\n{input_ids}\ninput_id_shape: {inputs['input_ids'].shape}\nattention_mask:\n{attention_mask}" +\
            f"\nlogits:\n{output_logits}, shape:\n{output_logits.shape}" +\
            f"\npositive_energy:\n{pos_energy.shape}\nnegative_energy:\n{neg_energy.shape}" +\
            f"\nargmax_logit:\n{torch.argmax(output_logits, dim=-1)}" +\
            f"\nmax_logit:\n{output_logits.gather(dim=-1, index=torch.argmax(output_logits, dim=-1).unsqueeze(-1)).squeeze(-1)}"
        )
        
        return (loss, outputs) if return_outputs else loss
    

    def train_step(self, train_loader):
        
        progress_bar = tqdm(range(len(train_loader)), desc="Training", disable=not self.accelerator.is_local_main_process)
        avg_loss = 0.
        
        self.model.train()
        for _, batch in enumerate(train_loader):
            with self.accelerator.accumulate(self.model):
                
                loss = self.compute_loss(
                    model=self.model, 
                    inputs=batch,
                )
                
                avg_loss += loss.item()
                
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.accelerator.log({"gradient_norm": grad_norm.mean()})
                    self.accelerator.log({"avg_loss": avg_loss / self.accelerator.gradient_accumulation_steps})
                    avg_loss = 0.

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.3f}")

            self.accelerator.log({"learning_rate": self.lr_scheduler.get_last_lr()[0]})
            self.accelerator.log({"update_step": _})
                  
        release_memory()
            
    def input_text_process(self, input_texts):
        return input_texts 
        
    def get_scores_from_texts(self, input_texts, mode='sum_logits'):
        input_texts = self.input_text_process(input_texts)
        inputs = self.tokenizer(
                input_texts, 
                return_tensors="pt", 
                add_special_tokens=self.config["add_special_tokens"], 
                padding=True,
                truncation=True,
            ).to(self.accelerator.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        self.model.eval()
        with torch.no_grad(): 
            outputs = self.model(**inputs)
        
            # (B, seq_len, V)
            output_logits = outputs.get("logits")
            output_log_probs = torch.log_softmax(output_logits.float(), dim=-1)
            
            
            if self.mode == 'generation':
                answer_start_from = get_answer_start_idx(input_ids, self.answer_token)
                answer_ids = input_ids[:, answer_start_from:] 
                answer_mask = attention_mask[:, answer_start_from:] 
                
                total_len = input_ids.shape[1]
                
                # (B, seq_len)
                logits = output_logits.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1) * answer_mask
                log_probs = output_log_probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1) * answer_mask    
            
                if mode == 'sum_logits':
                    return logits.sum(dim=-1).detach()
                if mode == 'mean_logits':
                    return logits.mean(dim=-1).detach()
                if mode == 'log_prob':
                    return log_probs.sum(dim=-1).detach()
                if mode == 'neg_ppl':
                    return - torch.exp(-log_probs.sum(dim=-1)/total_len).detach()
            
            if self.mode == 'classification':
                return output_logits.detach().squeeze(-1)
        
        
        
    
