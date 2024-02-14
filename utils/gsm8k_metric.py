import re
from utils.loggers import loggers
import numpy as np


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def parse_last_number(input_str):
    # Pattern to match numbers with optional decimal points and commas
    pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"

    matches = re.findall(pattern, input_str)
    if matches:
        # Remove commas and convert to number format
        last_match = matches[-1].replace(",", "")
        # If the number is a decimal ending in 0 (e.g., 90.00), convert to an integer
        if last_match.endswith('.00'):
            return str(int(float(last_match)))
        else:
            return last_match

    return None

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        # Remove commas for proper number conversion
        match_str = match_str.replace(",", "")
        # Handle decimal numbers ending in .00
        if match_str.endswith('.00'):
            return str(int(float(match_str)))
        else:
            return match_str
    else:
        return parse_last_number(completion)

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion.strip().strip('.')) == gt_answer


def get_accuracy(results):
    # Initialize a dictionary to store round-wise accuracies
    round_accuracies = {}

    for ans, gt, round in zip(results["completions"], results["ground_truths"], results["rounds"]):
        correct = is_correct(ans, gt)
        if round not in round_accuracies:
            round_accuracies[round] = {'correct': 0, 'total': 0}
        round_accuracies[round]['correct'] += correct
        round_accuracies[round]['total'] += 1

        accuracy_so_far = round_accuracies[round]['correct'] / round_accuracies[round]['total']
        loggers["eval"].info(f"\n{'-'*20}\ncompletion:\n{ans}\nground-truth:\n{gt}\nround: {round}\ncorrect: {correct}, round accuracy so far: {accuracy_so_far}")

    # Calculating overall accuracy and standard deviation
    overall_accuracy = sum([round_acc['correct'] for round_acc in round_accuracies.values()]) / sum([round_acc['total'] for round_acc in round_accuracies.values()])
    round_wise_accuracies = [round_acc['correct'] / round_acc['total'] for round_acc in round_accuracies.values()]
    std_dev = np.std(round_wise_accuracies)

    # Logging final results
    loggers["eval"].info(f"{'='*20}\nOverall Accuracy: {overall_accuracy}\nStandard Deviation across rounds: {std_dev}")

    # Return both overall accuracy and standard deviation
    return overall_accuracy, std_dev

def stop_criterion(input_text):
    last_sentence = input_text.strip().split("\n")[-1]
    if any(x in last_sentence for x in ["####", "\n\n"]):
        return True
    return False
    
    
    