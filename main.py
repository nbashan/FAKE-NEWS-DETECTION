from classifier.runner import runner
from setup.setup import setup

if __name__ == '__main__':
    for i in range(5):
        runner(ngram_range=2,max_features_start=1000,max_features_jump=1000,max_features_end=3000,ngram=False,outpath="try" + str(i) + ".xlsx").get_excel_results()
