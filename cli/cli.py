from classifier.textClassification import TextClassification

def generic_cli_():
    #def __init__(self,path: str, sheet_name_train: str,sheet_name_dev:str, sheet_name_test:str,feature_columns: list[int],label_column: int):
    path = input("path of excel database")
    sheet_name_train = input("sheet name of training set")
    sheet_name_dev = input("sheet name of dev set or -1")
    sheet_name_test = input("sheet name of test set or -1")
    print("##################### FEATURE COLUMNS TO END ENTER -1")
    feature_columns = []
    i = eval(input("feature column:"))
    while i != -1:
        feature_columns.append(i)
        i = eval(input("feature column:"))
    label_column  = eval(input("label column:"))
    tc = TextClassification(path,sheet_name_train,sheet_name_dev,sheet_name_test,feature_columns,label_column)
    print("OPTIONS :\n"
          "1: get score\n"
          "2: save classifier\n"
          "3: print options\n"
          "4: EXIT\n"
          "DON'T WORRY MORE TO COME!!\n")
    while True:
        choice = eval(input("please enter a choice:\n"))
        if choice == 1:
            type = input("dev or test?:\n")
            while type != "dev" and type != "test":
                type = input("ERROR!! dev or test?:\n")

            print("METHOD OPTIONS:\n"
                  "svc for LinearSVC()\n"
                  "rf for RandomForestClassifier()\n"
                  "mlp for MLPClassifier()\n"
                  "lr for LogisticRegression\n"
                  "mnb for MultinomialNB\n"
                  "lnr for LinearRegression\n")
            method = input("method: ")
            while (
                    method != "svc" and method != "rf" and method != "mlp" and method != "lr" and method != "mnb" and method != "lnr"):
                method = input("method: ")

            print("SCORE: ", tc.get_score(type, method))

        elif choice == 2:
            print("METHOD OPTIONS:\n"
                  "svc for LinearSVC()\n"
                  "rf for RandomForestClassifier()\n"
                  "mlp for MLPClassifier()\n"
                  "lr for LogisticRegression\n"
                  "mnb for MultinomialNB\n"
                  "lnr for LinearRegression\n")
            method = input("method: ")
            while (
                    method != "svc" and method != "rf" and method != "mlp" and method != "lr" and method != "mnb" and method != "lnr"):
                method = input("method: ")
            tc.save_classifier(method)
        elif choice == 3:
            print("OPTIONS :\n"
                  "1: get score\n"
                  "2: save classifier\n"
                  "3: print options\n"
                  "4: EXIT\n"
                  "DON'T WORRY MORE TO COME!!\n")
        elif choice == 4:
            print("BYE BYE!!!")
            break
        else:
            print("ERROR")