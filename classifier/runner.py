import xlsxwriter

from setup.setup import setup


class runner:
    def __init__(self,ngram_range:int,max_features_start:int,max_features_jump,max_features_end:int,outpath:str = "out3.xlsx",ngram:bool = True):
        self.__methods = ["rf", "abc","l_svc", "mlp", "lr", "mnb", "svc", "dtc", "knr"]
        self.__outpath = outpath
        self.__ngram = ngram
        self.__ngram_range = ngram_range
        self.__max_features_start = max_features_start
        self.__max_features_jump = max_features_jump
        self.__max_features_end = max_features_end

    def get_excel_results(self):
        exelWrite = xlsxwriter.Workbook(self.__outpath)
        outSheet = exelWrite.add_worksheet()
        row = 0
        if self.__ngram:
            ngram_range_list = [(x, y) for x in range(1,self.__ngram_range) for y in range(x, 6)]
        else:
            ngram_range_list = [(x, x) for x in range(1,self.__ngram_range)]
        for max_features in range(self.__max_features_start, self.__max_features_end + 1, self.__max_features_jump):
            for ngram_range in ngram_range_list:
                row += 1
                tc = setup(max_features=max_features,
                           ngram_range=ngram_range)
                tc.write_to_excel(methods=tc.get_results(testOrDev="test", *self.__methods),
                                  row=row,
                                  out_sheet=outSheet)
                print(row)
        exelWrite.close()