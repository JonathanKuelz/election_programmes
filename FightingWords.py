from collections import Counter
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from tqdm import tqdm


class Corpus:

    def __init__(self, terms: List[str], weights: Union[np.array, List[int]]):
        self.terms = terms
        self.weights = np.asarray(weights)
        self.n = self.weights.sum()

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as text_file:
            lines = text_file.read().split('\n')
        terms, weights = zip(*sorted(list(line.split(': ') for line in lines if not line == ''))[::-1])
        return cls(terms, list(map(int, weights)))

    @classmethod
    def empty(cls):
        return Corpus(list(), list())

    @property
    def dictionary(self):
        return dict(zip(self.terms, self.weights))

    def counts(self, words: List[str]):
        D = self.dictionary
        return np.fromiter((D.get(word, 0) for word in tqdm(words)), 'float32', count=len(words))

    def __len__(self):
        return len(self.terms)

    def __add__(self, other):
        if not isinstance(other, Corpus):
            return NotImplemented
        counter = Counter()
        counter.update(self.dictionary)
        counter.update(other.dictionary)
        terms, weights = zip(*sorted(list(counter.items()), key=lambda x: x[1]))
        return Corpus(terms, weights)


class FightingWords:
    """
    Adapted from the convoKit Fighting Words class: https://github.com/jmhessel/FightingWords
    Compare extract to corpus and get extract fighting words.
    """

    def __init__(self, corpus: Corpus, extract: Corpus, prior: Union[Dict[str, Union[int, float]], int, float]):
        self.corpus = corpus
        self.extract = extract
        self.vocabulary = list(set(self.corpus.terms).union(set(self.extract.terms)))
        if isinstance(prior, int) or isinstance(prior, float):
            self.prior = np.ones(len(self.vocabulary), dtype=float) * prior
        elif isinstance(prior, dict):
            self.prior = np.fromiter((prior.get(word, 0) + 1 for word in self.vocabulary), 'float32',
                                     count=len(self.vocabulary))
        else:
            raise NotImplementedError('Currently, prior is only supported as dictionary or as a constant.')
        self.count_matrix = np.stack([corpus.counts(self.vocabulary), extract.counts(self.vocabulary)])
        self.z_scores = None

    def _bayes_compare_language(self):
        """Details: p.13ff http://languagelog.ldc.upenn.edu/myl/Monroe.pdf"""
        vocab_size = len(self.vocabulary)
        print('Corpus Size: {}\tExtract Size: {}\tVocabulary size: {}'.format(
            len(self.corpus), len(self.extract), vocab_size))
        a0 = np.sum(self.prior)
        n1 = float(self.corpus.n)
        n2 = float(self.extract.n)
        n = np.asarray([n1, n2])
        M = self.count_matrix  # Unpacking for readable code
        terms = np.log((M + self.prior) / (n[:, np.newaxis] + a0 - M - self.prior))
        deltas = terms[0] - terms[1]
        # Approximate Variance (I guess based in Taylor Expansion)
        var = 1. / (M[0] + self.prior) + 1. / (M[1] + self.prior)
        self.z_scores = deltas / np.sqrt(var)

    def get_ngram_scores(self):
        if self.z_scores is None:
            print('Calculating z-scores...')
            self._bayes_compare_language()
        return pd.DataFrame(data=zip(*[self.z_scores, self.vocabulary]), columns=('z-score', 'ngram')) \
            .set_index('ngram') \
            .sort_values('z-score')

    def get_top_k_ngrams(self, top_k: int = 25, corpus: str = 'first') -> Tuple[List[str], List[float]]:
        assert corpus in ('first', 'second')
        order = 1 if (corpus == 'first') else -1
        if self.z_scores is None:
            self._bayes_compare_language()
        indices = np.argsort(self.z_scores)[::order][:top_k]
        return np.asarray(self.vocabulary)[indices], np.asarray(self.z_scores)[indices]

    def get_zscore(self, ngram):
        if ngram not in self.vocabulary:
            raise ValueError("Term not in Vocabulary")
        return self.z_scores[self.vocabulary.index(ngram)]

    def common_fighting_words(self, other: 'FightingWords') -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        common_vocabulary = list(set(self.vocabulary).intersection(set(other.vocabulary)))
        own_scores = self.z_scores[np.asarray([self.vocabulary.index(t) for t in common_vocabulary])]
        other_scores = other.z_scores[np.asarray([other.vocabulary.index(t) for t in common_vocabulary])]
        positive_mask = (own_scores > 0) * (other_scores > 0)
        negative_mask = (own_scores < 0) * (other_scores < 0)
        positive_scores = own_scores[positive_mask] * other_scores[positive_mask]
        negative_scores = own_scores[negative_mask] * other_scores[negative_mask]
        pos_sort = np.argsort(positive_scores)
        neg_sort = np.argsort(negative_scores)
        p_voc = np.asarray(common_vocabulary)[positive_mask][pos_sort]
        n_voc = np.asarray(common_vocabulary)[negative_mask][neg_sort]
        return list(zip(p_voc, positive_scores[pos_sort])), list(zip(n_voc, negative_scores[neg_sort]))
