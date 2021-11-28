import operator
import pickle
import pandas as pd
import scipy.sparse.csr
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np
import xlsxwriter.worksheet



# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
from classifier.skipGramVectorizer import SkipGramVectorizer

pd.set_option("display.max_rows", None, "display.max_columns", None)
class TextClassification:
    def __init__(self, path: str, sheet_name_train: str, sheet_name_dev: str, sheet_name_test: str,
                 feature_columns: list[int], label_column: int,
                 max_features: int = 1000, ngram_range: tuple = (1, 1), analyzer: str = "word",jump:int = -1,
                 min_df: int = 1, fold: bool = False) -> None:
        self.__max_features = max_features
        self.__ngram_range = ngram_range
        self.__analyzer = analyzer
        self.__jump = jump
        self.__fold= fold

        # if self.__jump == -1:
        self.__vectorizer = TfidfVectorizer(ngram_range=self.__ngram_range,
                                                analyzer=self.__analyzer,
                                                stop_words='english',lowercase=False, min_df=min_df)
        # elif self.__jump == 0:
        #     self.vectorizer = CountVectorizer(analyzer='word', stop_words='english',lowercase=False)
        # else:
        #     self.vectorizer = SkipGramVectorizer(max_features=max_features, analyzer=self.__analyzer, n=ngram_range[0], k=self.__jump, lowercase=False)
        # self.unfited = False
        self.__tfidfTransformer = TfidfTransformer()
        self.__methods = {
            "l_svc": LinearSVC(),
            "rf": RandomForestClassifier(),
            "mlp": MLPClassifier(),
            "lr": LogisticRegression(),
            "mnb": MultinomialNB(),
            "lnr": LinearRegression(),
            "knr": KNeighborsClassifier(),
            "svc": SVC(),
            "gpc": GaussianProcessClassifier(),
            "dtc": DecisionTreeClassifier(),
            "abc": AdaBoostClassifier(),
            "gnb": GaussianNB(),
            "qda": QuadraticDiscriminantAnalysis()
        }
        self.__method_saved = {
            "l_svc": "empty", "rf": "empty",
            "mlp": "empty", "lr": "empty",
            "mnb": "empty", "lnr": "empty",
            "knr": "empty", "svc": "empty",
            "gpc": "empty", "dtc": "empty",
            "abc": "empty", "gnb": "empty",
            "qda": "empty"
        }
        self.__path = path
        self.__sheet_name_train = sheet_name_train
        self.__sheet_name_dev = sheet_name_dev
        self.__sheet_name_test = sheet_name_test
        self.__feature_columns = feature_columns
        self.__label_column = label_column
        self.__X_train, self.__y_train = self.__init_X_y("train", self.__sheet_name_train)
        self.__X_dev = self.__y_dev = self.__X_test = self.__y_test = None


    def get_results(self, *methods, testOrDev:str) -> dict:
        if not methods:
            methods = self.__method_saved.keys()
        results = {}
        for method in methods:
            results[method] = self.__get_score(testOrDev, method)
        return results

    def write_to_excel(self,methods: dict, row, out_sheet: xlsxwriter.worksheet.Worksheet) -> None:
        if row == 1:
            j = 1
            for method in methods:
                out_sheet.write(0, j, method)
                j += 1
        j = 0
        out_sheet.write(row, j, f"{self.__ngram_range}gram\n max features = {self.__max_features}\ntype: word")
        for method in methods:
            j += 1
            out_sheet.write(row, j, methods[method])

    def save_classifier(self, method: str) -> None:
        new_path = method + '.pickle'
        self.__method_saved[method] = new_path
        clf = self.__methods['lnr']
        clf.fit(self.__X_train, np.ravel(self.__y_train))
        with open(new_path, 'wb') as f:
            pickle.dump(clf, f)

    def get_sorted_unique(self):
        vectorizer = CountVectorizer(analyzer='word', stop_words='english', lowercase=True)
        array_of_words = []
        df_train = np.array(pd.read_excel(open(self.__path, 'rb'), sheet_name=self.__sheet_name_train))
        features = [df_train[:, [feature]] for feature in self.__feature_columns]
        row1, temp1 = features[0].shape
        for row in range(row1):
            sentence = ""
            for data in features:
                sentence += data[row][0]
            array_of_words.append(sentence)
        a = array_of_words[0]
        for sentence in array_of_words:
            a = a + " " + sentence
        a.lower()
        a = [a]
        count_wm = vectorizer.fit_transform(a)
        count_tokens = vectorizer.get_feature_names()
        dcount = {}
        df_countvect = pd.DataFrame(data=count_wm.toarray(), index=['count'], columns=count_tokens).to_numpy()
        for word in df_countvect[0]:
            if word in dcount:
                dcount[word] +=1
            else:
                dcount[word] = 1
        dcount =  sorted(dcount.items(), key=operator.itemgetter(0))
        greaterThan3 = 0
        for item in dcount:
            a,b = item
            if a >=3:
                greaterThan3 += b
        print(greaterThan3)


    def __get_score(self, testOrDev: str, method: str) -> float:
        if testOrDev == "dev" and self.__X_dev is None:
            self.__X_dev, self.__y_dev = self.__init_X_y(testOrDev, self.__sheet_name_dev)
        if testOrDev == "test" and self.__X_test is None:
            if self.__fold:
                self.__X_train,self.__X_test,self.__y_train,self.__y_test = train_test_split(self.__X_train,self.__y_train, test_size = 0.25)
            else:
                self.__X_test, self.__y_test = self.__init_X_y(testOrDev, self.__sheet_name_test)
            return self.__get_clf(method).score(X=self.__X_test, y=self.__y_test)
        if testOrDev == "dev":
            return self.__get_clf(method).score(X=self.__X_dev, y=self.__y_dev)
        if testOrDev == "test":
            return self.__get_clf(method).score(X=self.__X_test, y=self.__y_test)

    def __get_clf(self, method: str) -> sklearn:
        if self.__method_saved[method] != "empty":
            clf = open(method, 'rb')
            pickle.load(clf)
        else:
            clf = self.__methods[method]
            clf.fit(self.__X_train, np.ravel(self.__y_train))
        return clf

    def __get_features(self, features, testOrDev) -> scipy.sparse.csr.csr_matrix:
        array_of_words = []
        row1, temp1 = features[0].shape
        for row in range(row1):
            sentence = ""
            for data in features:
                sentence += data[row][0]
            array_of_words.append(sentence)
        if testOrDev == "train":
            x = self.__vectorizer.fit_transform(array_of_words)
            # if self.__jump >0:
            y = self.__tfidfTransformer.fit_transform(x.toarray())
            # else:
            #     return x
        else:
            x = self.__vectorizer.transform(array_of_words)
            # if self.__jump > 0:
            y = self.__tfidfTransformer.transform(x.toarray())
            # else:
            #     return x
        return y

    def __init_X_y(self, testOrDev, sheet_name) -> (scipy.sparse.csr.csr_matrix, np.ndarray):
        df_train = np.array(pd.read_excel(open(self.__path, 'rb'), sheet_name=sheet_name))
        features = [df_train[:, [feature]] for feature in self.__feature_columns]
        return self.__get_features(features, testOrDev), df_train[:, [self.__label_column]].astype('int')