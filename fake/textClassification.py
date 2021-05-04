import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import numpy as np


class TextClassification:
    def __init__(self, path: str, sheet_name_train: str, sheet_name_dev: str, sheet_name_test: str,
                 feature_columns: list[int], label_column: int):
        self.__tfidfTransformer = TfidfTransformer()
        self.__tfidfVectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer='word', stop_words='english')
        self.__methods = {
            "svc": LinearSVC(),  # 0.821078431372549
            "rf": RandomForestClassifier(),  # 0.8284313725490197
            "mlp": MLPClassifier(),  # 0.7843137254901961
            "lr": LogisticRegression(),  # 0.7549019607843137
            "mnb": MultinomialNB(),  # 0.7549019607843137
            "lnr": LinearRegression()  # 0.45570619772827226
        }
        self.__method_saved = {
            "svc": "empty",
            "rf": "empty",
            "mlp": "empty",
            "lr": "empty",
            "mnb": "empty",
            "lnr": "empty"
        }
        self.__path = path
        self.__sheet_name_train = sheet_name_train
        self.__sheet_name_dev = sheet_name_dev
        self.__sheet_name_test = sheet_name_test
        self.__feature_columns = feature_columns
        self.__label_column = label_column
        self.__X_train,self.__y_train = self.__init_X_y("train", self.__sheet_name_train)
        self.__X_dev = self.__y_dev = self.__X_test = self.__y_test = None

    def get_score(self, testOrDev: str, method: str):
        if testOrDev == "dev" and self.__X_dev is None:
            self.__X_dev, self.__y_dev = self.__init_X_y(testOrDev, self.__sheet_name_dev)
        if testOrDev == "test" and self.__X_test is None:
            self.__X_test, self.__y_test = self.__init_X_y(testOrDev, self.__sheet_name_test)
            return self.__get_clf(method).score(X=self.__X_test, y=self.__y_test)
        if testOrDev == "dev":
            return self.__get_clf(method).score(X=self.__X_dev, y=self.__y_dev)
        if testOrDev == "test":
            return self.__get_clf(method).score(X=self.__X_test, y=self.__y_test)

    def save_classifier(self, method: str) -> None:
        new_path = method + '.pickle'
        self.__method_saved[method] = new_path
        clf = self.__methods['lnr']
        clf.fit(self.__X_train, np.ravel(self.__y_train))
        with open(new_path, 'wb') as f:
            pickle.dump(clf, f)

    def __get_clf(self, method: str):
        if self.__method_saved[method] != "empty":
            clf = open(method, 'rb')
            pickle.load(clf)
        else:
            clf = self.__methods[method]
            clf.fit(self.__X_train, np.ravel(self.__y_train))
        return clf

    def __get_features(self, features, testOrDev):
        array_of_words = []
        row1, temp1 = features[0].shape
        for row in range(row1):
            sentence = ""
            for data in features:
                sentence += data[row][0]
            array_of_words.append(sentence)
        if testOrDev == "train":
            x = self.__tfidfVectorizer.fit_transform(array_of_words)
            y = self.__tfidfTransformer.fit_transform(x.toarray())
        else:
            x = self.__tfidfVectorizer.transform(array_of_words)
            y = self.__tfidfTransformer.transform(x.toarray())
        return y

    def __init_X_y(self, testOrDev, sheet_name):
        df_train = np.array(pd.read_excel(open(self.__path, 'rb'), sheet_name=sheet_name))
        features = [df_train[:, [feature]] for feature in self.__feature_columns]
        return self.__get_features(features, testOrDev), df_train[:, [self.__label_column]].astype('int')
