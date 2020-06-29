import numpy as np

def initial_resample(x, y, res_type='all'):
    """
    function takes x and y
    then trying to find single best resample
    return res_x, res_y
    """
    from imblearn.combine import SMOTEENN, SMOTETomek
    resamplers = []
    rand = 42

    if   res_type == 'under':
        pass
    elif res_type == 'over':
        pass
    elif res_type == 'combined':
        resamplers.append(('SMOTEENN', SMOTEENN(random_state=rand)))
        resamplers.append(('SMOTETomek', SMOTETomek(random_state=rand)))
    elif res_type == 'all':
        pass

    x,y = 1,1
    return x ,y



def encode_y_ELM_binary(y_input):
    # current ELM realisation requires 1 and -1 as 'y' values
    # TODO use other ELM package, or rewrite current
    y = y_input.copy()
    for i in range(len(y)):
        if y[i] == y[0]:
            y[i] = 1
        else:
            y[i] = -1
    return y.astype(np.int8)





##################################################################

class DataPreprocessing:

    def __init__(self, DS, num_index, categ_index):  # txt?

        self.DS = DS.copy()

        self.num_index = num_index
        self.categ_index = categ_index

        self.num_col = self.DS[:, self.num_index]
        self.categ_col = self.DS[:, self.categ_index]

    def encode_cat_col(self):  # TODO pandas to numpy
        from category_encoders import OrdinalEncoder
        enc = OrdinalEncoder(return_df=False).fit(self.categ_col)
        self.categ_col = enc.transform(self.categ_col)

        # DEBUG
        print(self.DS)
        print(self.categ_col)
        # return pandas, IDK why

    def preproc_txt(self):
        # TODO
        pass

    def get_x(self):
        # if cat col exist encode
        if len(self.categ_index) != 0:

            self.encode_cat_col()

            if len(self.num_index) != 0:
                print('has Num, has Categ')
                x = np.hstack([self.num_col, self.categ_col])
            else:
                print('no Num, has Categ')
                x = self.categ_col

        else:
            print('no Categ, has Num')
            x = self.num_col

        return x.astype(float)
