import os
import warnings
warnings.simplefilter('ignore')

from transformers import logging
logging.set_verbosity_error()

import torch
import numpy as np

import argparse
from utils.util import load_config
from accelerate.utils import set_seed
from datasets import load_dataset


def experiment(config):
    if config['task'].lower() == 'truthfulqa':
        from algo.task_adapters.truthfulqa_adapter import TruthfulQA_Adapter as Adapter
        from algo.task_adapters.truthfulqa_adapter import PROMPT
        aif_dataset_path = config.get("aif_dataset_path", None)
        dataset = load_dataset('truthful_qa', 'generation', split="validation").shuffle(seed=config["seed"])
        train_dataset, test_dataset = dataset.train_test_split(train_size=config["train_ratio"], shuffle=False).values()
        prompt = PROMPT

    elif config['task'].lower() == 'gsm8k':
        from algo.task_adapters.gsm8k_adapter import GSM8K_Adapter as Adapter
        from algo.task_adapters.gsm8k_adapter import PROMPT
        train_dataset = load_dataset('gsm8k', 'main', split="train").shuffle(seed=config["seed"])
        test_dataset = load_dataset('gsm8k', 'main', split="test").shuffle(seed=config["seed"])
        prompt = PROMPT 
        
    elif config['task'].lower() == 'strategyqa':
        from algo.task_adapters.strategyqa_adapter import StrategyQA_Adapter as Adapter
        from algo.task_adapters.strategyqa_adapter import PROMPT, PROMPT_NO_INST
        dataset = load_dataset('wics/strategy-qa', split="test").shuffle(seed=config["seed"])
        train_dataset, test_dataset = dataset.train_test_split(train_size=config["train_ratio"], shuffle=False).values()
        prompt = PROMPT if config.get("use_prompt_instruction", True) else PROMPT_NO_INST
        
    elif config['task'].lower() =='scienceqa':
        from algo.task_adapters.scienceqa_adapter import ScienceQA_Adapter as Adapter
        from algo.task_adapters.scienceqa_adapter import PROMPT, PROMPT_NO_INST
        train_dataset = load_dataset("derek-thomas/ScienceQA", split="train").shuffle(seed=config["seed"])
        train_dataset = train_dataset.filter(lambda x: x['image'] == None).select(range(2000))
        test_dataset = load_dataset("derek-thomas/ScienceQA", split="test").shuffle(seed=config["seed"])
        test_dataset = test_dataset.filter(lambda x: x['image'] == None).select(range(500))
        prompt = PROMPT if config.get("use_prompt_instruction", True) else PROMPT_NO_INST
        
    else:
        raise NotImplementedError

    adapter = Adapter(
        prompt=prompt, 
        config=config,
        )
    adapter.train(
        train_dataset=train_dataset, 
        test_dataset=test_dataset,
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/truthfulqa.yaml")
    args = parser.parse_args()
    
    config_path = args.config
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"
    
    config = load_config(config_path)
    
    # set seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    set_seed(config["seed"])
    
    experiment(config)