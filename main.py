import warnings

from classifier.runner import runner
from setup.setup import setup
warnings.filterwarnings('ignore')


if __name__ == '__main__':
        runner(ngram_range=4,
               max_features_start=1000,
               max_features_jump=1000,
               max_features_end=5000,
               ngram=False,
               outpath="finalDataset" + ".xlsx").get_excel_results()
