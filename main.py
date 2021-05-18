from classifier.runner import runner
from setup.setup import setup

if __name__ == '__main__':
    runner(ngram_range=5,max_features_start=1000,max_features_jump=1000,max_features_end=11000,ngram=False).get_excel_results()
