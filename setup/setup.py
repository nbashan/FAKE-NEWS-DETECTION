from classifier.textClassification import TextClassification

path = "OUR_DIVIDES.xlsx"
sheet_name_train = "train"
sheet_name_dev = "dev"
sheet_name_test = "test"
feature_columns =[0,1]
label_column = 2

def setup(max_features):
    return TextClassification(path,sheet_name_train,sheet_name_dev,sheet_name_test,feature_columns,label_column,max_features)
