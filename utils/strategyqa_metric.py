import numpy as np
from utils.loggers import loggers


def is_correct(model_completion, gt_answer):
    # Extract the boolean value from the model completion.
    # Assuming the format is "xxxx\nxxxx\nxxxx\n #### Yes." or "xxxx\nxxxx\nxxxx\n #### No."
    model_answer = model_completion.split('####')[-1].strip().lower().replace('.', '')

    # Check if the extracted answer is valid
    if model_answer not in ['yes', 'no']:
        return False

    # Convert the extracted answer to boolean
    model_answer_bool = True if model_answer == 'yes' else False

    return model_answer_bool == gt_answer


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
    
    
    