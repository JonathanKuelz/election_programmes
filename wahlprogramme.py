from collections import Counter
from functools import reduce
import operator
from pathlib import Path
import math
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from FightingWords import Corpus, FightingWords
from typing import Dict, List
import re
from wordcloud import WordCloud


TITLES = {
    'afd': 'AfD',
    'cdu_csu': 'CDU/CSU',
    'fdp': 'FDP',
    'gruene': 'BÜNDNIS 90/DIE GRÜNEN',
    'linke': 'DIE LINKE',
    'spd': 'SPD'
}


def preprocess(txt: str) -> List[str]:
    remove_words = ['bundestagswahlprogramm', 'f', 'ff', 'kapitel', 'seite']
    german_stop_words = stopwords.words('german') + remove_words
    alpha = re.sub('[\\W_0-9]+', ' ', txt).lower()
    return [tok for tok in alpha.split(' ') if tok not in german_stop_words]
    # return [tok for tok in alpha.split(' ') if tok not in remove_words]


def ngrams(tokens: List[str], n):
    if n == 1:
        return tokens
    grams = []
    for i in range(len(tokens) - n + 1):
        grams.append(' '.join(tokens[i:i+n]))
    return grams


def fighting_words_against_all_others(corpora: Dict[str, Corpus], key: str):
    self_corpus = corpora[key]
    other_corpus = reduce(operator.add, (corpora[k] for k in corpora if k != key))
    return FightingWords(self_corpus, other_corpus, prior=0.1)


def draw_fighting_words(fw: Dict[str, FightingWords], save: Path) -> None:
    for name, fw_object in fw.items():
        wc = WordCloud(background_color="white", width=1920, height=1080, min_font_size=8)
        cloud_data = dict(zip(*fw_object.get_top_k_ngrams(100, 'second')))
        wc.generate_from_frequencies({w: math.sqrt(weight) for w, weight in cloud_data.items()})
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(TITLES[name], loc='center')
        # Set path to save below, parent folder needs to be created manually (or you need to add a line of code here)
        plt.savefig(save.joinpath('choose_your_folder').joinpath(name), dpi=500)
        print('Min z-score for {}: {}'.format(name, min(cloud_data.values())))


def main():
    path = Path('data/')
    files = list(path.iterdir())
    programs = {p.name.split('.')[0]: preprocess(p.open('r').read()) for p in files if p.is_file()}
    for party, content in programs.items():
        print("Das Programm der {} enthält {} Wörter.".format(party, len(content)))
    # Adapt n-gram length below
    words_and_n_grams = {party: ngrams(content, n) for n in range(1, 2) for party, content in programs.items()}
    corpora = dict()
    for party, grams in words_and_n_grams.items():
        cnt = Counter(grams)
        terms, weights = list(zip(*cnt.items()))
        corpora[party] = Corpus(terms, weights)
    fighting_words = {party: fighting_words_against_all_others(corpora, party) for party in corpora}
    draw_fighting_words(fighting_words, path)


if __name__ == '__main__':
    main()
