import numpy as np
from category_encoders import OrdinalEncoder


class DataPreprocessing:
    def __init__(self, DS, num_index, categ_index):  # txt?
        self.DS = DS.copy()
        self.num_index = num_index
        self.categ_index = categ_index
        self.num_col = self.DS[:, self.num_index]
        self.categ_col = self.DS[:, self.categ_index]

    def encode_cat_col(self):
        enc = OrdinalEncoder(return_df=False).fit(self.categ_col)
        self.categ_col = enc.transform(self.categ_col)
        # DEBUG
        print(self.DS)
        print(self.categ_col)

    def get_x(self):
        # if cat col exist, then encode
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
