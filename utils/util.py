import yaml
import torch
from collections import Counter
import re
from datasets import load_dataset, Dataset

EXCEPTION_LIST = ['Mr', 'Mrs', 'Ms', 'Dr', 'Sr', 'Jr', 'Prof', 'St', 'Mt', 'Ft', 'No', 'no', 'etc', \
    'i.e', 'e.g', 'cf', 'cf.', 'Fig', 'fig', 'Figs', 'figs', 'Vol', 'vol', 'Vols', 'vols', \
    'Ch', 'ch', 'Sec', 'sec', 'Secs', 'secs', 'Eq', 'eq', 'Eqs', 'eqs', 'Fig', 'fig', 'Figs', 'figs', \
    'Ref', 'ref', 'Refs', 'refs', 'App', 'app', 'Apps', 'apps', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec', \
    'vs', 'Vs', 'etc', 'i.e', 'e.g', 'U.S', 'U.K', 'U.A.E', 'P.R.C', 'U.S.A', 'J.R.R']

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def format_end2end_prompt(q, ans, info=False):

    """Formats prompt for fine-tuned end-to-end truth/info scores with GPT-3"""

    if info:
        prompt = 'Q: {0}\nA: {1}\nHelpful:'.format(q, ans)
    else:
        prompt = 'Q: {0}\nA: {1}\nTrue:'.format(q, ans)
    return prompt

def extract_first_sentences(text_list):
    """Extracts the first sentence from a list of text."""
    first_sentences = []
    
    for text in text_list:
        # Splitting at each period
        text = text.strip().split('\n')[0]
        parts = text.split('. ')
        first_sentence = ''
        for part in parts:
            if not first_sentence:
                first_sentence = part
            else:
                # Check if the previous part ended with an abbreviation or a number
                if any(first_sentence.endswith(abbreviation) for abbreviation in EXCEPTION_LIST):
                    first_sentence += '. ' + part
                elif first_sentence[-1].isdigit():
                    first_sentence += '. ' + part
                else:
                    break
        first_sentence += '.' if not first_sentence.strip().endswith('.') else ''
        first_sentences.append(first_sentence)

    return first_sentences

def deduplication(text_list, num_to_keep, fill_to):
    if len(text_list) == 0:
        raise ValueError("The list of texts is empty.")
    # Count the frequency of each item
    freq_count = Counter(text_list)

    # Create a set of unique items
    unique_items = list(set(text_list))

    # Sort the unique items by their frequency in descending order
    unique_items.sort(key=lambda x: -freq_count[x])

    # Append items to make the list of at least 'num_to_keep' length, prioritizing higher frequency items
    while len(unique_items) < num_to_keep:
        for item in unique_items:
            if len(unique_items) >= num_to_keep:
                break
            unique_items.append(item)

    # fill to the end of the list with "<EMPTY>"
    while len(unique_items) < fill_to:
        unique_items.append("<EMPTY>")
    return unique_items

def accumulate_strings(string_list):
    accumulated_list = []

    for s in string_list:
        lines = s.split('\n')
        accumulated = []
        current_accumulation = ""
        for line in lines:
            current_accumulation += line if not current_accumulation else '\n' + line
            accumulated.append(current_accumulation)
        accumulated_list.extend(accumulated)

    return accumulated_list

def get_answer_start_idx(full_tensor, pattern):
    
    # long_tensor = full_tensor.clone().cpu().flip(dims=[0])
    long_tensor = full_tensor.detach().cpu().flip(dims=[0])
    pattern = pattern.cpu().flip(dims=[0])
    len_pattern = pattern.shape[0]
    len_long_tensor = long_tensor.shape[0]

    # Iterate over the larger tensor
    for i in range(len_long_tensor - len_pattern + 1):
        # Extract a slice of the larger tensor
        window = long_tensor[i:i+len_pattern]

        # Check if this slice matches the pattern
        if torch.equal(window, pattern):
            # Return the index of the last element of the matched sequence
            return len_long_tensor - i
    return 0


def chunk_input_texts(input_texts, chunk_size=2):
    """Chunk the input_texts into smaller parts."""
    for i in range(0, len(input_texts['truthful']), chunk_size):
        yield {
            'truthful': input_texts['truthful'][i:i + chunk_size],
            'informative': input_texts['informative'][i:i + chunk_size]
        }

def build_pubmed_subset(dataset, M, min_sentence_count=0, max_sentence_count=100, tolerance=30):
    # Function to check the sentence count in the long answer
    def is_valid_long_answer(long_answer):
        sentence_count = long_answer.count('. ') + long_answer.count('! ') + long_answer.count('? ') + 1
        return min_sentence_count <= sentence_count <= max_sentence_count

    # Initialize counters and a list for the subset
    yes_count, no_count = 0, 0
    subset = []

    # Iterate over the dataset
    for sample in dataset:
        if is_valid_long_answer(sample['long_answer']):
            if sample['final_decision'] == 'yes' and yes_count - no_count < tolerance:
                subset.append(sample)
                yes_count += 1
            elif sample['final_decision'] == 'no' and no_count - yes_count < tolerance:
                subset.append(sample)
                no_count += 1

        if len(subset) >= M:  
            break

    return Dataset.from_list(subset)

def split_demo_dataset(dataset, num_shots):
    # Initialize lists to store indices
    yes_indices = []
    no_indices = []

    # Iterate through the dataset to find balanced 'yes' and 'no' samples
    for i, sample in enumerate(dataset):
        if len(yes_indices) < num_shots // 2 and sample['final_decision'] == 'yes':
            yes_indices.append(i)
        elif len(no_indices) < num_shots // 2 and sample['final_decision'] == 'no':
            no_indices.append(i)
        
        # Break if enough samples are collected
        if len(yes_indices) == num_shots // 2 and len(no_indices) == num_shots // 2:
            break

    # Extract the samples
    extracted_indices = yes_indices + no_indices
    extracted_samples = dataset.select(extracted_indices)

    # Create a new dataset without the extracted samples
    remaining_dataset = dataset.filter(lambda example, index: index not in extracted_indices, with_indices=True)

    return extracted_samples, remaining_dataset

# Function to count words
def word_count(text):
    return len(text.split())

# Function to filter the dataset
def filter_length_for_alpaca(example):
    total_length = word_count(example['instruction']) + word_count(example['input']) + max(word_count(example['output_1']), word_count(example['output_2']))
    return total_length < 500

def extract_answer(input_string):
    # Find the index of "A:"
    answer_index = input_string.find("Q:")
    # Check if "A:" is found in the string
    if answer_index != -1:
        # Extract everything after "A:"
        prev_ans = input_string[answer_index:].strip()
    else:
        # If "A:" is not found, return an empty string or an error message
        prev_ans = ""
    return prev_ans