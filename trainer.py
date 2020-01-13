from jax import jit, grad
import jax.experimental.optimizers as optim
from dataloader import NumpyLoader
from model import SkipGramEmbeddings
from sgns_loss import SGNSLoss
from tqdm import tqdm
from datasets.pypi_lang import PyPILangDataset
from datasets.world_order import WorldOrderDataset
from functools import partial
import numpy as np


class Trainer:

    def __init__(self, args):
        # Load data
        self.args = args
        self.dataset = PyPILangDataset(args)#, examples_path='data/pypi_examples.pth', dict_path='data/pypi_dict.pth')
        self.vocab_size = len(self.dataset.dictionary)
        print("Finished loading dataset")

        self.dataloader = NumpyLoader(self.dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers)

        self.model = SkipGramEmbeddings(self.vocab_size, args.embedding_len)
        self.sgns = SGNSLoss(self.dataset)
        # Set up optimizer - rmsprop seems to work the best
        self.opt_init, self.opt_update, self.get_params = optim.adam(args.lr)

    @partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.sgns.forward)(params, batch)
        return self.opt_update(i, g, opt_state), params, g

    def train(self):
        # Initialize optimizer state!
        opt_state = self.opt_init(self.model.word_embeds)
        for epoch in range(self.args.epochs):
            print(f'Beginning epoch: {epoch + 1}/{self.args.epochs}')
            for i, batch in enumerate(tqdm(self.dataloader)):
                opt_state, params, g = self.update(i, opt_state, batch)
            self.log_step(epoch, params, g)

    def log_step(self, epoch, params, g):
        print(f'EPOCH: {epoch} | GRAD MAGNITUDE: {np.sum(g)}')
        # Log embeddings!
        print('\nLearned embeddings:')
        for word in self.dataset.queries:
            print(f'word: {word} neighbors: {self.model.nearest_neighbors(word, self.dataset.dictionary, params)}')
