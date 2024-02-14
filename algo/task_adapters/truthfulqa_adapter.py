from algo.adapter import Adapter
from utils.truthfulqa_metric import get_accuracy, stop_criterion
from utils.util import format_end2end_prompt
from utils.loggers import loggers

import wandb
from concurrent.futures import TimeoutError
from algo.beam_search import Beam_Search
from copy import copy
from accelerate.utils import gather_object
from tqdm.auto import tqdm

PROMPT = '''
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
'''.strip() + "\n"

class TruthfulQA_Adapter(Adapter):
    
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        self.token_usage = {"input": 0, "output": 0}
        
        super().__init__(
                config=config,
                prompt=prompt,
            )
        self.acc_table = wandb.Table(columns=["stage", "truthful", "informative", "overall"]) 

    def prepare_for_training(self, batch, dataset_path="", use_adapter=True):
        
        def process_batch_item(b):
            try:
                question = b['question']
                beam_search = Beam_Search(
                            params=self.config,
                            thought_generator=self.thought_generator,
                            init_sequence=question,
                            stop_criterion=stop_criterion,
                            qa_template=self.config["qa_template"]
                        )
                negative_ans = beam_search(return_with_init=False)[:self.config["num_negatives_for_training"]]
                if negative_ans is None:
                    return None
                negative_ans = list(set(negative_ans)) # simple deduplication
                negative_ans = [ans.strip().replace("_", "") for ans in negative_ans]
                neg_texts = [self.config["qa_template"].replace("<Q>", question).replace("<A>", ans) for ans in negative_ans]
                return neg_texts
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
            results = dict(negative_texts=[])
            for idx in batch_idx:
                b = batch[idx]
                result = process_batch_item(b)
                if result:
                    results["negative_texts"].extend(result)
                progress_bar.update(self.accelerator.num_processes)
            results = [results]
            
        gathered_results = gather_object(results)
        if self.accelerator.is_main_process:  
            positive_texts = []
            negative_texts = []
            
            for b in batch:
                question = b['question']
                positive_ans = self.get_positive_ans(b)
                positive_ans = positive_ans if isinstance(positive_ans, list) else [positive_ans]
                positive_texts.extend([self.config["qa_template"].replace("<Q>", question).replace("<A>", ans) for ans in positive_ans])
                # positive_texts.extend([f"Q: {question}\n\nA: {ans}" for ans in positive_ans])

            for result in gathered_results:
                negative_texts.extend(result["negative_texts"])
            
            if self.config["use_dataset_negative_ans"]:
                incorrect_ans = [ans[0] if isinstance(ans, list) else ans for ans in b['incorrect_answers']]
                positive_texts.extend([self.config["qa_template"].replace("<Q>", question).replace("<A>", ans) for ans in incorrect_ans])
            
            self.build_dataset(
                    positive_texts, 
                    negative_texts, 
                    save_to=dataset_path
                )


    def evaluate(self, eval_dataset, use_adapter=True, stage_name=""):
        
        def process_batch_item(b):
            try:
                question = b['question']
                if use_adapter:
                    beam_search = Beam_Search(
                        params=self.config,
                        thought_generator=self.thought_generator,
                        init_sequence=question,
                        stop_criterion=stop_criterion,
                        qa_template=self.config["qa_template"],
                    )
                    answer = beam_search(return_with_init=False)
                else:
                    # prompt = f"{self.prompt}\nQ: {question}\n\nA: "
                    qa_text = self.config["qa_template"].replace("<Q>", question).replace("<A>", "")
                    prompt = f"{self.prompt}\n{qa_text}"
                    answer = self.generator.get_response(prompt, n=1, stop=["\n\n"], max_tokens=100, extract_first_sentence=True)
                    loggers["api"].info(f"\n{'='*20}\nQuery:\n{prompt}\nResponses:\n{answer}")

                if answer is None:
                    return None

                return question, answer[0]
            
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
            results = dict(truthful=[], informative=[], rounds=[])
            for idx, round in zip(splitted_dict["list_idx"], splitted_dict["round_idx"]):
                b = eval_dataset[idx]
                result = process_batch_item(b)
                if result:
                    question, answer = result
                    results["truthful"].append(format_end2end_prompt(question, answer, info=False))
                    results["informative"].append(format_end2end_prompt(question, answer, info=True))
                    results["rounds"].append(round)
                progress_bar.update(self.accelerator.num_processes)
                    
            results = [results]
            
        gathered_results = gather_object(results)
        
        if self.accelerator.is_main_process:  
            results = dict(truthful=[], informative=[], rounds=[])
            for result in gathered_results:
                results["truthful"].extend(result["truthful"])
                results["informative"].extend(result["informative"])
                results["rounds"].extend(result["rounds"])
            accuracy, std = get_accuracy(results)
            stats_text = "\n".join(f"{k}: {accuracy[k]*100:.2f} ± {std[k]*100:.2f}" for k in ["overall", "truthful", "informative"])
            print(f"\nStage: {stage_name}, {stats_text}")
            data = [stage_name] + [accuracy[k] for k in ["truthful", "informative", "overall"]]
            self.acc_table.add_data(*data)
            self.accelerator.log({"accuracy": copy(self.acc_table)})
        
        
    def get_positive_ans(self, b):
        positive_ans = [b['best_answer']] + [ans[0] if isinstance(ans, list) else ans for ans in b['correct_answers']]
        return positive_ans



    

