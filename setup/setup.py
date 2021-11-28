from classifier.textClassification import TextClassification

#OUR_DIVIDES.xlsx
#data_set_1.xlsx
def setup(max_features: int = 1000, ngram_range: tuple = (1, 1), analyzer: str = "word",path = "OUR_DIVIDES.xlsx") -> TextClassification:
    return TextClassification(path=path,
                              sheet_name_train="train",
                              sheet_name_dev="dev",
                              sheet_name_test="test",
                              feature_columns=[0,1],
                              label_column=2,
                              max_features=max_features,
                              ngram_range=ngram_range,
                              analyzer=analyzer,
                              jump=0,
                              min_df=4,
                              fold= True)
