import random
from abc import ABC

'''
===================================================
Abstract Guesser Class
===================================================
'''
class Guesser(ABC):
    def __init__(self, min_level, max_level):
        '''
        Initialises the Guesser.

        Args:
            min_level (int): the lowest (least complex) level in the LLM chain.
            max_level (int) the highest (most complext) level in the LLM chain, excluding GPT-4o (the highest level, which we want to avoid using).
        
        Returns:
            None
        '''
        self.min_level = min_level
        self.max_level = max_level

    def guess(self, nl_query):
        '''
        Guesses the complxity level of a given natural langauge query. The complexirty level is the minimum level on the LLM stack atthe Guesser thinks is able to correctly answer the query.

        Args:
            nl_query (str): the query to be posed to the LLM chain, expressed in natural language.
        
        Returns:
            int: the earliest (minimum) level of the LLM chain the Guesser thinks that this query can be answered at.
        '''
        raise NotImplementedError("This function has not been implemented")

    def train_model(self, *args, **kwargs):
        '''
        Trains the Guesser.

        Args:
            None (accepts anything as a catch-all to encompass various APIs).

        Returns:
            tuple: the guesser model
        '''
        pass

    def eval_model(self, *args, **kwargs):
        '''
        Evaluates the Guesser.

        Args:
            None (accepts anything as a catch-all to encompass various APIs).

        Returns:
            dict of str -> float: a dictionary of evalaution statistics
        '''
        pass
    
    def save(self, path):
        '''
        Saves this Guesser model to disk.

        Args:
            path: the path to save the guesser model.

        Returns:
            tuple: the guesser model, and a path to its save location on disk
        '''
        pass

    def load(self, path):
        '''
        Loads this Guesser model from disk.

        Args:
            path: the path to the guesser model.

        Returns:
            tuple: the guesser model, and a path to its save location on disk
        '''
        pass

'''
===================================================
Baseline Guessers
===================================================
'''
class MinGuesser(Guesser):
    def guess(self, nl_query):
        return self.min_level

class AvgGuesser(Guesser):
    def guess(self, nl_query):
        return int((self.min_level + self.max_level) / 2)
    
class RandGuesser(Guesser):
    def guess(self, nl_query):
        return int(
            (random.random() * (1 + self.max_level - self.min_level)) + self.min_level
        )

'''
===================================================
ML-based Guessers
===================================================
'''
class ROBERTA_Guesser(Guesser):
    '''
    What is needed:
        We need to fine-tune ROBERTA to classify onto our LLM stack. In order to do this, we need data examples of where to start. What I am thinking is basically a 2-column CSV of <query>,<level> (though in practice, this is likely easier to store in json).

        Query should be the NLP query posed. Level should be the first level at which the LLM chain produced an **executable** or **coerect** query (whether it was correct or not).
            - note -- do we want to say its the first level we got a correct query at? Using the Spider2SPARQL benchmark (https://arxiv.org/html/2309.16248v2) we know the right answers -- and we can just promt up the chain until we get a correct one. I like that better, lets go with that.
            - but get data on both, so can choose to guess correct or guess executable.
            - we should use the BLEU measure to say yes if the predicted query is X% correct compared tyo the ground truth query. For the exact percent, we should look at the dist, then choose the cuttoff from there. Matt has an implementation of BLEU in oldProject/.
            - for useful examples, see: "validate_sparql_syntax(query)" and "check_query_execution(query, endpoint="https://dbpedia.org/sparql")" in oldProject/analysis.ipynb

    For ROBERTA, use reference: https://pytorch.org/hub/pytorch_fairseq_roberta/
    '''
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def guess(self, nl_query):
        return self.min_level

    def train_model(self):
        raise NotImplementedError("Not implemented")

    def eval_model(self):
        raise NotImplementedError("Not implemented")
