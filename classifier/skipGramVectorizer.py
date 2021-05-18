import nltk
from sklearn.feature_extraction.text import CountVectorizer


class SkipGramVectorizer(CountVectorizer):

    def __init__(self, n, k, **kwargs):
        super(SkipGramVectorizer, self).__init__(**kwargs)
        self.n = n
        self.k = k

    def skip_analyzer(self, doc):
        tokens = super().build_analyzer()(doc)
        if self.n <= 1:
            return nltk.ngrams(tokens, n=self.n)
        return nltk.skipgrams(tokens, self.n, self.k)

    def build_analyzer(self):
        return self.skip_analyzer
