import jax.numpy as np
import jax.nn as nn
from utils import AliasMultinomial


class SGNSLoss:
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 15

    def __init__(self, dataset):
        super(SGNSLoss, self).__init__()
        self.dataset = dataset
        self.vocab_len = len(dataset.dictionary)

        # Helpful values for unigram distribution generation
        # Should use cfs instead but: https://github.com/RaRe-Technologies/gensim/issues/2574
        self.transformed_freq_vec = np.array([dataset.dictionary.dfs[i] for i in range(self.vocab_len)]) ** self.BETA
        self.freq_sum = np.sum(self.transformed_freq_vec)
        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, params, batch):
        # Unpack data
        center_ids, context_ids = batch
        # Get vectors
        center, context = params[center_ids], params[context_ids]
        # Squeeze into dimensions we want
        center, context = center.squeeze(), context.squeeze()  # batch_size x embed_size

        # Compute true portion
        true_scores = (center * context).sum(-1)  # batch_size
        loss = self.bce_loss_w_logits(true_scores, np.ones_like(true_scores))

        # Compute negatively sampled portion - NUM_SAMPLES # of negative samples for each true context
        for i in range(self.NUM_SAMPLES):
            samples = self.get_unigram_samples(n=center.shape[0], word_embeds=params)
            neg_sample_scores = (center * samples).sum(-1)
            # Update loss
            loss += self.bce_loss_w_logits(neg_sample_scores, np.zeros_like(neg_sample_scores))

        return loss

    @staticmethod
    def bce_loss_w_logits(x, y):
        max_val = np.clip(x, 0, None)
        loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
        return loss.mean()

    def get_unigram_samples(self, n, word_embeds):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idxs = self.unigram_table.draw(n)
        return word_embeds[rand_idxs].squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf)
