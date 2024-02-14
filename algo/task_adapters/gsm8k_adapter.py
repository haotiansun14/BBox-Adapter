from algo.reasoning_adapter import Reasoning_Adapter
from utils.gsm8k_metric import get_accuracy, stop_criterion, is_correct

# imported from https://github.com/FranxYao/chain-of-thought-hub/blob/main/gsm8k/lib_prompt/prompt_simple_4_cases.txt
PROMPT = '''
Q: Ivan has a bird feeder in his yard that holds two cups of birdseed. Every week, he has to refill the emptied feeder. Each cup of birdseed can feed fourteen birds, but Ivan is constantly chasing away a hungry squirrel that steals half a cup of birdseed from the feeder every week. How many birds does Ivan’s bird feeder feed weekly?
A: Let's think step by step.
The squirrel steals 1/2 cup of birdseed every week, so the birds eat 2 - 1/2 = 1 1/2 cups of birdseed.
Each cup feeds 14 birds, so Ivan’s bird feeder feeds 14 * 1 1/2 = 21 birds weekly.
#### The answer is 21

Q: Samuel took 30 minutes to finish his homework while Sarah took 1.3 hours to finish it. How many minutes faster did Samuel finish his homework than Sarah?
A: Let's think step by step.
Since there are 60 minutes in 1 hour, then 1.3 hours is equal to 1.3 x 60 = 78 minutes.
Thus, Samuel is 78 – 30 = 48 minutes faster than Sarah.
#### The answer is 48

Q: Julia bought 3 packs of red balls, 10 packs of yellow balls, and 8 packs of green balls. There were 19 balls in each package. How many balls did Julie buy in all?
A: Let's think step by step.
The total number of packages is 3 + 10 + 8 = 21.
Julia bought 21 × 19 = 399 balls.
#### The answer is 399

Q: Lexi wants to run a total of three and one-fourth miles. One lap on a particular outdoor track measures a quarter of a mile around. How many complete laps must she run?
A: Let's think step by step.
There are 3/ 1/4 = 12 one-fourth miles in 3 miles.
So, Lexi will have to run 12 (from 3 miles) + 1 (from 1/4 mile) = 13 complete laps.
#### The answer is 13
'''.strip() + "\n"

class GSM8K_Adapter(Reasoning_Adapter):
    
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
        answer = b['answer'].replace("####", "#### The answer is")
        positive_ans = [answer]
        return positive_ans
    
    
    def formulate_question(self, b):
        return b['question']

    
    def extract_ground_truth(self, b):
        return b
    
    

