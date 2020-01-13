import jax
import torch
import numpy as onp


class SkipGramEmbeddings:

    def __init__(self, vocab_size, embed_len):
        super(SkipGramEmbeddings, self).__init__()
        # This initialization is important!
        torch_embed = torch.nn.Embedding(vocab_size, embed_len)
        self.word_embeds = jax.numpy.array(torch_embed.weight.detach().numpy())

    def forward(self, center, context):
        """
        Acts as a lookup for the center and context words' embeddings

        :param center: The center words indicies
        :param context: The context words indicies
        :return: The embedding parameters
        """
        return self.word_embeds[center], self.word_embeds[context]

    @staticmethod
    def nearest_neighbors(word, dictionary, vectors):
        """
        Finds vector closest to word_idx vector
        :param word: String
        :param dictionary: Gensim dictionary object
        :return: Integer corresponding to word vector in self.word_embeds
        """
        index = dictionary.token2id[word]
        query = vectors[index]

        ranks = vectors.dot(query).squeeze()
        denom = query.T.dot(query).squeeze()
        denom = denom * onp.sum(vectors ** 2, 1)
        denom = onp.sqrt(denom)
        ranks = ranks / denom
        mostSimilar = []
        [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
        nearest_neighbors = mostSimilar[:10]
        nearest_neighbors = [dictionary[comp] for comp in nearest_neighbors]

        return nearest_neighbors

