import torch
from utils.loggers import loggers


class Beam_Search():
    
    def __init__(self, params, thought_generator, init_sequence, stop_criterion=None, qa_template="Q: <Q>\n\nA: <A>"):
        self.beam_size = params['beam_size']
        self.num_candidates = params['num_candidates']
        self.max_length = params['max_length']
        self.early_stopping = params['early_stopping'] 
    
        self.thought_generator = thought_generator
        self.stop_criterion = stop_criterion
        
        self.qa_template = qa_template
        
        self.step = 0
        
        self.sequences = {
            hash(''): '',
            'init_seq': init_sequence
        }
        
        # store the scores for each beam 
        self.scores = torch.zeros(self.beam_size)
        # store the seq_keys for each beam
        self.beams = - torch.ones((self.beam_size, self.max_length), dtype=torch.long)
        
        
    def get_strings_from_ids(self, batched_ids):
        if batched_ids == 'init_seq':
            return [self.sequences['init_seq']]
        batched_ids = batched_ids.numpy().astype(int)
        return ['\n'.join([self.sequences[id] for id in seq_ids]) for seq_ids in batched_ids]
    
    
    def add_sequence(self, input_texts):
        candidate_seq_ids = torch.zeros(self.num_candidates, dtype=torch.long)
        for i, text in enumerate(input_texts):
            key = hash(text)
            self.sequences[key] = text
            candidate_seq_ids[i] = key
        return candidate_seq_ids
        
    
    def get_beam_strings(self, return_with_init=True):
        if self.step == 0:
            return [self.qa_template.replace('<Q>', self.get_strings_from_ids('init_seq')[0]).replace('<A>', '')]
        current_beam_strings = self.get_strings_from_ids(self.beams[:, :self.step])
        if return_with_init:
            return [self.qa_template.replace('<Q>', self.get_strings_from_ids('init_seq')[0]).replace('<A>', f"{string}\n") for string in current_beam_strings]
        return current_beam_strings
        
        
    def get_candidates(self, current_string, current_score):
        if not self.stop_criterion(current_string) or self.step == 0:
            # decode for the current_ids
            res = self.thought_generator(current_string) # (num_candidates,)
            
            if res == '<SKIP>':
                return '<SKIP>'

            seq_ids = self.add_sequence(res['text'])
            scores = res['scores']
            return dict(
                seq_ids=seq_ids,
                scores=scores,
            )
        else:
            seq_ids = hash('') * torch.ones(self.num_candidates)
            scores = current_score * torch.ones(self.num_candidates)
            scores[1:] = -float('inf')
            return dict(
                seq_ids=seq_ids,
                scores=scores,
            )


    def beam_step(self):
        current_string = self.get_beam_strings(return_with_init=True) # (beam_size,), (beam_size,)
        bs = 1 if self.step == 0 else self.beam_size
        next_ids = torch.zeros((bs * self.num_candidates), dtype=torch.long)
        next_scores = torch.zeros(bs * self.num_candidates)
        which_beam = - torch.ones((bs * self.num_candidates), dtype=torch.long)
        for i in range(bs):
            # get the seq_ids and scores for the i-th candidate in the current beam
            candidates = self.get_candidates(current_string[i], self.scores[i])
            
            if candidates == '<SKIP>':
                return '<SKIP>'
            
            # stack the seq_ids and aggregate the scores
            next_ids[i*self.num_candidates:(i+1)*self.num_candidates] = candidates['seq_ids']
            next_scores[i*self.num_candidates:(i+1)*self.num_candidates] = candidates['scores']
            which_beam[i*self.num_candidates:(i+1)*self.num_candidates] = i
        
        # sort the next scores and get the indices
        top_scores, top_indices = next_scores.topk(self.beam_size, sorted=True)
        beam_ids = which_beam[top_indices]
        self.beams = self.beams[beam_ids, :]
        self.beams[:, self.step] = next_ids[top_indices]
        self.scores = top_scores
        
        # update the step
        self.step += 1

        sp_text = '\n\n'.join([f"{t} (Score: {s}, Parent: {id})" for t, s, id in zip(self.get_beam_strings(return_with_init=False), self.scores, beam_ids)])
        loggers["search"].info(f"{'*'*10} Step: {self.step} {'*'*10}\n\n{sp_text}\n")
        
        if self.early_stopping and all(id in [hash(''), hash('.')] for id in self.beams[:, self.step-1]):
            return '<EARLY_STOP>'
        return '<CONTINUE>'
        
        
    def __call__(self, return_with_init=False):
        '''
        perform beam search
        '''
        loggers["search"].info(f"{'='*20}\nQ: {self.get_strings_from_ids('init_seq')[0]}\n\n")

        for _ in range(self.max_length):
            flag = self.beam_step()
        
            # early stopping
            if flag == '<EARLY_STOP>':
                break
            
            # skip question
            if flag == '<SKIP>':
                loggers["error"].info(f"Beam search timed out for {self.get_strings_from_ids('init_seq')[0]}")
                return None
            
        sp_text = '\n\n'.join([f"{t} (Score: {s})" for t, s in zip(self.get_beam_strings(return_with_init=False), self.scores)])
        loggers["search"].info(f'{"-"*10} FINAL {"-"*10}\nstep {self.step}\n{sp_text}\n{self.beams}\n')
        return self.get_beam_strings(return_with_init)
        
            

        
 