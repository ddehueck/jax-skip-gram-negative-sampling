import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import cycle
from gensim.corpora import Dictionary
from torch.utils.data.dataset import Dataset


class SkipGramDataset(Dataset):

    def __init__(self, args, fname='data/pypi_nodes_lang.csv'):
        self.args = args
        self.fname = fname
        self.dictionary = None
        self.examples = []
        self.name = ''

    def _get_generator(self):
        for df in pd.read_csv(self.fname, header=None, chunksize=1):
            doc = self._tokenize(df.values)
            for example in self._generate_examples_from_file(doc):
                yield example

    def _tokenize(self, doc):
        return doc.split()

    def __iter__(self, index):
        return cycle(self._get_generator())

    def _build_dictionary(self):
        """
        Creates a Gensim Dictionary
        :return: None - modifies self.dictionary
        """
        print("Building Dictionary...")
        self.dictionary = Dictionary(self.load_files())

    def _generate_examples_from_file(self, file):
        """
        Generate all examples from a file within window size
        :param file: File from self.files
        :returns: List of examples
        """

        examples = []
        for i, token in enumerate(file):
            if token == -1:
                # Out of dictionary token
                continue

            # Generate context tokens for the current token
            context_words = self._generate_contexts(i, file)

            # Form Examples:
            # center, context - follows form: (input, target)
            new_examples = [(token, ctxt) for ctxt in context_words if ctxt != -1]

            # Add to class
            examples.extend(new_examples)
        return examples

    def _generate_contexts(self, token_idx, tokenized_doc):
        """
        Generate Token's Context Words
        Generates all the context words within the window size defined
        during initialization around token.

        :param token_idx: Index at which center token is found in tokenized_doc
        :param tokenized_doc: List - Document broken into tokens
        :returns: List of context words
        """
        contexts = []
        # Iterate over each position in window
        for w in range(-self.args.window_size, self.args.window_size + 1):
            context_pos = token_idx + w

            # Make sure current center and context are valid
            is_outside_doc = context_pos < 0 or context_pos >= len(tokenized_doc)
            center_is_context = token_idx == context_pos

            if is_outside_doc or center_is_context:
                # Not valid - skip to next window position
                continue

            contexts.append(tokenized_doc[context_pos])
        return contexts

    def _example_to_tensor(self, center, target):
        """
        Takes raw example and turns it into tensor values

        :params example: Tuple of form: (center word, document id)
        :params target: String of the target word
        :returns: A tuple of tensors
        """
        center, target = np.array([int(center)]), np.array([int(target)])
        return center, target
