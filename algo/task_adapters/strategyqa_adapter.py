from algo.reasoning_adapter import Reasoning_Adapter
from utils.strategyqa_metric import get_accuracy, stop_criterion, is_correct

PROMPT = '''
Use the step-by-step method as shown in the examples to answer the question. Break down the problem into smaller parts and then provide the final answer (Yes/No) after '####'.

Example 1:
Q: Karachi was a part of Alexander the Great's success?

A: Karachi is a city in modern day Pakistan.
Krokola was an ancient port located in what is now Karachi.
Alexander the Great stationed his fleet in Krokola on his way to Babylon.
Alexander the Great defeated Darius and conquered Babylon before expanding his 
empire.
#### Yes.

Example 2:
Q: Was P. G. Wodehouse's favorite book The Hunger Games?

A: P. G. Wodehouse died in 1975.
The Hunger Games was published in 2008.
#### No.

Your Question:
'''.strip()

PROMPT_NO_INST = '''
Q: Karachi was a part of Alexander the Great's success?

A: Karachi is a city in modern day Pakistan.
Krokola was an ancient port located in what is now Karachi.
Alexander the Great stationed his fleet in Krokola on his way to Babylon.
Alexander the Great defeated Darius and conquered Babylon before expanding his 
empire.
#### Yes.

Q: Was P. G. Wodehouse's favorite book The Hunger Games?

A: P. G. Wodehouse died in 1975.
The Hunger Games was published in 2008.
#### No.
'''.strip() + "\n"

class StrategyQA_Adapter(Reasoning_Adapter):
    
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        
        super().__init__(
                config=config,
                prompt=prompt,
            )
 
        self.stop_criterion = stop_criterion
        self.get_accuracy = get_accuracy
        self.is_correct = is_correct
        self.qa_template = config["qa_template"]
        
        
    def get_positive_ans(self, b):
        positive_ans = '\n'.join(b['facts']) + '\n#### '
        positive_ans += 'Yes.' if b['answer'] else 'No.'
        return positive_ans if isinstance(positive_ans, list) else [positive_ans]
    
    
    def formulate_question(self, b):
        return b['question']

    def extract_ground_truth(self, b):
        return b['answer']
    
    

