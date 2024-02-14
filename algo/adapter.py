from tqdm.auto import tqdm
from llms.api import LLM_API
from llms.whitebox import Whitebox_LLM
from utils.util import load_config, deduplication, extract_answer
from utils.loggers import loggers, update_log_folder
from accelerate.utils import release_memory
from datasets import load_from_disk
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime


class Adapter(Whitebox_LLM):
    '''
    The Adapter is used to achieve the post-adaptation via
    - generate raw output from black-box LLMs
    - evaluate the output using white-box LLMs
    - use the eval scores and beam search to get the final output
    '''
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        self.token_usage = {"input": 0, "output": 0}

        # create generator
        self.generator = LLM_API(
            model=self.config['generator_model'], 
            query_params=self.config
        )
        
        # create adapter
        super().__init__(
            config=config,
            )
        
        self.accelerator.init_trackers(
            project_name=self.config['task'], 
            config=self.config | {"learning_rate": self.optimizer.param_groups[0]["lr"]},
            init_kwargs={
                "wandb": {
                    "save_code": True, 
                    "name": datetime.now().strftime("%Y%m%d-%H%M%S")
                }
            }
        )
                
                
    def thought_generator(self, input_string):
        '''
        The thought_generator is used to generate the next thoughts given the current CoT
        via combining generator and critic
        input: 
        [prompt]\nQ: xxx\n\nA: xxx"
        '''
        # get generated candidates for generator
        prompt = f"{self.prompt}\n{input_string}"
        prompt = prompt.strip()

        if self.config["max_length"] > 1 or "truthfulqa" in self.config["task"].lower():
            generated_texts = self.generator.get_response(prompt, max_tokens=self.config["max_tokens"], extract_first_sentence=True)
        elif self.config["task"].lower() == "alpacafarm":
            generated_texts = self.generator.get_response(prompt, max_tokens=self.config["max_tokens"], extract_first_sentence=False)
        else:
            generated_texts = self.generator.get_response(prompt, stop=["\n\n"], max_tokens=500, extract_first_sentence=False)
            
        if generated_texts == '<SKIP>' or len(generated_texts) < 1:
            return '<SKIP>'
        
        generated_texts = [t.strip() for t in generated_texts if t.strip() != '.']
        
        generated_texts = deduplication(
            generated_texts, 
            num_to_keep=self.config['beam_size'],
            fill_to=self.config['num_candidates']
        )
        if self.config.get("only_eval_answers", False):
            # eliminate the input_string before "A:" and save it as prev_ans
            prev_ans = extract_answer(input_string)
            texts_to_score = [prev_ans + t.strip() for t in generated_texts]
        else:
            texts_to_score = [input_string + t.strip() for t in generated_texts]

        scores = self.get_scores_from_texts(texts_to_score, mode=self.config['score_mode'])
        
        # find <EMPTY> in generated_texts and make the corresponding scores -inf
        for i, t in enumerate(generated_texts):
            if t == "<EMPTY>":
                scores[i] = -float('inf')

        pr_text = '\n'.join([r + f' (Score: {s})' for r, s in zip(generated_texts, scores.tolist())])
        loggers["api"].info(f"\n{'='*20}\nQuery:\n{prompt}\nResponses:\n{pr_text}")
        
        ad_text = '\n\n'.join([f"{r} (Score: {s})" for r, s in zip(texts_to_score, scores.tolist())])
        loggers["adaptor"].info(f"\n{'='*20}\n{ad_text}")
        
        return {
            "text": generated_texts,
            "scores": scores
        }
            

    def train(self, train_dataset, test_dataset):
        torch.cuda.empty_cache()
        self.accelerator.free_memory()
        
        num_epochs = self.config['num_epochs']
        num_epochs_offline_warmup = self.config.get("num_epochs_offline_warmup", 1) 
        use_blackbox_warmup = self.config.get("use_blackbox_warmup", False)
        num_epochs_blackbox_warmup = self.config.get("num_epochs_blackbox_warmup", 1)
        num_online_finetuning_repeat = self.config.get("num_online_finetuning_repeat", 1)
        offline_paths = self.config.get("offline_warmup_path")
        
        # train-validation split
        validation_ratio = self.config.get("validation_ratio", 0.)
        if validation_ratio > 0 and test_dataset is None:
            train_dataset, eval_dataset = train_dataset.train_test_split(test_size=validation_ratio, shuffle=False).values()
        else: 
            eval_dataset = test_dataset
            
        self.accelerator.print(f"Train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")
        update_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, collate_fn=lambda x: x)
        
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
            )

        if self.config["eval_blackbox"]:
            update_log_folder(
                "eval_blackbox_only", 
                self.accelerator.process_index
                )
            self.evaluate(
                eval_dataset, 
                use_adapter=False,
                stage_name="Blackbox only"
            )
            self.accelerator.wait_for_everyone()
            
        if self.config["eval_unfinetuned"]:
            update_log_folder(
                "eval_raw_adapter", 
                self.accelerator.process_index
                )
            self.evaluate(
                eval_dataset, 
                use_adapter=True,
                stage_name="Raw adapter"
            )
            self.accelerator.wait_for_everyone()
            
        ### OFFLINE WARM-UP
        if type(offline_paths) == str:
            offline_paths = [offline_paths]
        self.accelerator.print(f"Offline warmup with: {offline_paths}")
        
        if offline_paths and all(os.path.isdir(d) for d in offline_paths):
            for i, dataset_path in enumerate(offline_paths):
                update_log_folder(
                    f"offline_warmup_{i}", 
                    self.accelerator.process_index
                    )
                with self.accelerator.main_process_first():
                    train_dataset = load_from_disk(dataset_path)
                
                train_loader = self.build_dataloader(train_dataset)
                
                progress_bar = tqdm(total=num_epochs_offline_warmup, desc="Offline Warmup", disable=not self.accelerator.is_local_main_process)

                for epoch in range(num_epochs_offline_warmup):     
                    self.train_step(train_loader)     
                    progress_bar.update(1)
                
                self.accelerator._dataloaders = []
                del train_loader
                release_memory()
                    
            self.evaluate(
                    eval_dataset, 
                    use_adapter=True,
                    stage_name=f"Eval: offline warmup"
                )
            self.accelerator.wait_for_everyone()

        ### ONLINE FINE-TUNING
        progress_bar = tqdm(total=num_epochs, desc="Training", disable=not self.accelerator.is_local_main_process)
        for epoch in range(num_epochs):
            for update_idx, update_batch in enumerate(update_dataloader):
                update_log_folder(
                    f"finetuning_epoch_{epoch}_idx_{update_idx}", 
                    self.accelerator.process_index
                    )
                
                if self.accelerator.is_main_process:
                    loggers["train"].info(f"\n{'='*20}\nepoch {epoch} | idx {update_idx}")
                    loggers["eval"].info(f"\n{'='*20}\nepoch {epoch} | idx {update_idx}")
                loggers["search"].info(f"\n{'='*20}\nepoch {epoch} | idx {update_idx}")
                loggers["tensor"].info(f"\n{'='*20}\nepoch {epoch} | idx {update_idx}")
                loggers["api"].info(f"\n{'='*20}\nepoch {epoch} | idx {update_idx}")
                
                dataset_path = f"data/{self.config['task']}/{self.config['critic_model']}/train_epoch_{epoch}_idx_{update_idx}"
                    
                if epoch < num_epochs_blackbox_warmup and use_blackbox_warmup:
                    self.prepare_for_training(update_batch, dataset_path, use_adapter=False)
                else:
                    self.prepare_for_training(update_batch, dataset_path, use_adapter=True)
                
                with self.accelerator.main_process_first():
                    train_dataset = load_from_disk(dataset_path)
                                    
                train_loader = self.build_dataloader(train_dataset)
                for _ in range(num_online_finetuning_repeat):
                    self.train_step(train_loader)    
                
                progress_bar.update(1)
                
            self.evaluate(
                eval_dataset, 
                use_adapter=True,
                stage_name=f"Eval: {epoch} | idx {update_idx}"
            )
            self.accelerator.wait_for_everyone()    
            
            self.accelerator._dataloaders = []
            del train_loader
            release_memory()
            
            self.report_token_usage()
        
        self.accelerator.wait_for_everyone()     
        self.accelerator.end_training()
                 
                    
    def report_token_usage(self):
        usage = self.generator.get_token_usage()
        self.generator.reset_token_usage()
        
        self.token_usage['input'] += usage['input']
        self.token_usage['output'] += usage['output']
        
        self.accelerator.log({"input_token": self.token_usage['input']})
        self.accelerator.log({"output_token": self.token_usage['output']})
        self.accelerator.log({"total_token": self.token_usage['input']+self.token_usage['output']})

        loggers["api"].info(
            f'Input tokens: {usage["input"]}/{self.token_usage["input"]}, output tokens: {usage["output"]}/{self.token_usage["output"]}, ' +\
                f'total tokens: {self.token_usage["input"] + self.token_usage["output"]}'
            )
    
    
    def prepare_for_training(self):
        raise NotImplementedError
    
    
    def evaluate(self):
        raise NotImplementedError
        
        
    def update_ground_truths(self, batch):
        pass
    
