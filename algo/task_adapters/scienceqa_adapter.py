from algo.reasoning_adapter import Reasoning_Adapter
from utils.scienceqa_metric import get_accuracy, stop_criterion, is_correct
from utils.util import EXCEPTION_LIST

PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. Respond to the question by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer after '####'.

Example:
Question: Which figure of speech is used in this text?
Dr. Shelton is unhappy with her new assistant because simple tasks, like fetching coffee, take him years to finish.
Choices: 
0: anaphora
1: hyperbole

Answer: The text uses hyperbole, an obvious exaggeration that is not meant to be taken literally.
Take him years to finish is an exaggeration, since it probably does not take him entire years to fetch coffee.
#### 1

Your Question:
'''.strip()

PROMPT_NO_INST = '''
Question: Which figure of speech is used in this text?
Dr. Shelton is unhappy with her new assistant because simple tasks, like fetching coffee, take him years to finish.
Choices: 
0: anaphora
1: hyperbole

Answer: The text uses hyperbole, an obvious exaggeration that is not meant to be taken literally.
Take him years to finish is an exaggeration, since it probably does not take him entire years to fetch coffee.
#### 1
'''.strip() + "\n"

class ScienceQA_Adapter(Reasoning_Adapter):
    
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
        positive_ans = [formulate_answer(b)]
        return positive_ans
    
    
    def formulate_question(self, b):
        question = b['question']
        choices = 'Choices: \n'
        choices += "\n".join([f"{i}: {b['choices'][i]}" for i in range(len(b['choices']))])
        return f"{question}\n{choices.strip()}"
    
    
    def extract_ground_truth(self, b):
        return b["answer"]
    
    
def formulate_answer(b):
    answer = b['answer']
    sol = b['solution']

    # Splitting manually, checking for exceptions
    solution_parts = []
    current_part = []
    words = sol.split()
    for word in words:
        # Add word to the current part
        current_part.append(word)
        # Check if the current word ends with a period and is not in the exception list
        if word.endswith('.') and not any(word.startswith(exc) for exc in EXCEPTION_LIST):
            # Join the current part into a sentence, add it to the solution parts, and reset current part
            solution_parts.append(' '.join(current_part))
            current_part = []
    # Add any remaining words as the last part
    if current_part:
        solution_parts.append(' '.join(current_part))

    # Join the solution parts with a newline for formatting
    solution = "\n".join(solution_parts).strip()

    return f"{solution}\n#### {answer}"
