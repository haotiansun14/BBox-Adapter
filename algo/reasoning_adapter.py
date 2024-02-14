from algo.adapter import Adapter
from utils.util import accumulate_strings
from utils.loggers import loggers
import wandb
from concurrent.futures import TimeoutError
from algo.beam_search import Beam_Search
from copy import copy
from accelerate.utils import gather_object
from tqdm.auto import tqdm

class Reasoning_Adapter(Adapter):
    
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        self.token_usage = {"input": 0, "output": 0}
        
        self.stop_criterion = None
        self.get_accuracy = None
        self.is_correct = None
        self.qa_template = None

        super().__init__(
                config=config,
                prompt=prompt,
            )
        self.acc_table = wandb.Table(columns=["stage", "accuracy"])
 
 
    def get_ans_from_blackbox(self, q, n=1, temp=1):
        qa_text = self.formulate_qa(q=q, a="")
        prompt = f"{self.prompt}\n{qa_text}"
        ans = self.generator.get_response(
                prompt, 
                n=n, 
                stop=["\n\n"], 
                max_tokens=500, 
                extract_first_sentence=False,
                temp=temp
            )
        loggers["api"].info(f"\n{'='*20}\nQuery:\n{prompt}\nResponses:\n{ans}")
        return ans
        

    def prepare_for_training(self, batch, dataset_path="", use_adapter=True):
        
        def process_batch_item(b):
            ground_truth = self.extract_ground_truth(b)
            try:
                question = self.formulate_question(b)
                if use_adapter:
                    beam_search = Beam_Search(
                                params=self.config,
                                thought_generator=self.thought_generator,
                                init_sequence=question,
                                stop_criterion=self.stop_criterion,
                                qa_template=self.qa_template
                            )
                    negative_ans = beam_search(return_with_init=False)[:self.config["num_negatives_for_training"]]
                else:
                    negative_ans = self.get_ans_from_blackbox(q=question, n=self.config["num_candidates_blackbox_warmup"])
                    
                if negative_ans is None:
                    return None
                
                negative_ans = list(set(negative_ans)) # simple deduplication
                negative_ans = [ans.strip() for ans in negative_ans]
                if self.config["only_eval_answers"]:
                    neg_texts = [ans for ans in accumulate_strings(negative_ans)]
                else:
                    neg_texts = [self.formulate_qa(q=question, a=ans) for ans in accumulate_strings(negative_ans)]
                return neg_texts, ground_truth
            
            except TimeoutError:
                loggers["error"].info("[prepare] Beam search timed out.")
                return None
            except Exception as e:
                loggers["error"].info(f"[prepare] Error in process_batch_item: {e}")
                return None
            
        list_idx = list(range(len(batch)))
        progress_bar = tqdm(total=len(list_idx), desc="prepare", disable=not self.accelerator.is_local_main_process)
        self.accelerator.wait_for_everyone()
        with self.accelerator.split_between_processes(list_idx) as batch_idx:
            results = dict(negative_texts=[], positive_texts=[])
            for idx in batch_idx:
                b = batch[idx]
                result = process_batch_item(b)
                if result:
                    completions, ground_truth = result
                    for c in completions:
                        if self.is_correct(c, ground_truth) and self.config["use_outcome_supervision"]:
                            results["positive_texts"].append(c)
                        else:
                            results["negative_texts"].append(c)
                progress_bar.update(self.accelerator.num_processes)
            results = [results]
            
        gathered_results = gather_object(results)
        if self.accelerator.is_main_process:  
            positive_texts = []
            negative_texts = []
            
            for b in batch:
                question = self.formulate_question(b)
                positive_ans = self.get_positive_ans(b)
                if self.config["only_eval_answers"]:
                    positive_texts.extend([ans for ans in accumulate_strings(positive_ans)])
                else:
                    positive_texts.extend([self.formulate_qa(q=question, a=ans) for ans in accumulate_strings(positive_ans)])

            for result in gathered_results:
                negative_texts.extend(result["negative_texts"])
                positive_texts.extend(result["positive_texts"])
                
            self.build_dataset(
                    positive_texts, 
                    negative_texts, 
                    save_to=dataset_path
                )


    def evaluate(self, eval_dataset, use_adapter=True, stage_name=""):
        
        def process_batch_item(b):
            ground_truth = self.extract_ground_truth(b)
            try:
                question = self.formulate_question(b)
                if use_adapter:
                    beam_search = Beam_Search(
                        params=self.config,
                        thought_generator=self.thought_generator,
                        init_sequence=question,
                        stop_criterion=self.stop_criterion,
                        qa_template=self.qa_template
                    )
                    answer = beam_search(return_with_init=False)
                else:
                    answer = self.get_ans_from_blackbox(q=question, n=1, temp=self.config["temperature"])
                if answer is None:
                    return None
                
                return answer[0], ground_truth
            
            except TimeoutError:
                loggers["error"].info("[eval] Beam search timed out.")
                return None
            
            except Exception as e:
                loggers["error"].info(f"[eval] Error in process_batch_item: {e}")
                return None
        
        split_dict = {
            "list_idx": list(range(len(eval_dataset))) * self.config["num_eval_rounds"],
            "round_idx": [i for i in range(self.config["num_eval_rounds"]) for _ in range(len(eval_dataset))]
        }
        
        progress_bar = tqdm(total=len(split_dict["list_idx"]), desc=stage_name, disable=not self.accelerator.is_local_main_process)
        
        self.accelerator.wait_for_everyone()
        with self.accelerator.split_between_processes(split_dict) as splitted_dict:
            results = dict(completions=[], ground_truths=[], rounds=[])
            for idx, round in zip(splitted_dict["list_idx"], splitted_dict["round_idx"]):
                b = eval_dataset[idx]
                result = process_batch_item(b)
                if result:
                    completion, ground_truth = result
                    results["completions"].append(completion)
                    results["ground_truths"].append(ground_truth)
                    results["rounds"].append(round)
                progress_bar.update(self.accelerator.num_processes)
            results = [results]
            
        gathered_results = gather_object(results)
        
        if self.accelerator.is_main_process:  
            results = dict(completions=[], ground_truths=[], rounds=[])
            for result in gathered_results:
                results["completions"].extend(result["completions"])
                results["ground_truths"].extend(result["ground_truths"])
                results["rounds"].extend(result["rounds"])
            accuracy, std = self.get_accuracy(results)
            print(f"\nStage: {stage_name}, Accuracy: {accuracy * 100:.2f}% ± {std * 100:.2f}%")
            data = [stage_name] + [accuracy]
            self.acc_table.add_data(*data)
            self.accelerator.log({"accuracy": copy(self.acc_table)})
        

    def formulate_qa(self, q, a):
        return self.qa_template.replace("<Q>", q).replace("<A>", a)


    def get_positive_ans(self, b):
        pass
    
    
    def formulate_question(self, b):
        pass


    def extract_ground_truth(self, b):
        pass

    

