from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from imblearn.under_sampling import AllKNN, RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter


def resample_data(x, y, num_features, cat_features, rtype):
    """
    res_type: str
     'under'    - Under-sampling
     'over'     - Over-sampling
     'combined' - Over-sampling combined with under-sampling
     'auto'     - try to automatically find the best resampling strategy
     self.x, self.y are returned after resampling
    """
    rand = 42
    numeric = len(num_features) != 0
    categorical = len(cat_features) != 0

    dist = list(Counter(y).items())
    print('Class distribution before resampling:', dist)

    n_row, n_col = x.shape
    if (n_row <= n_col) & (rtype == 'under'):
        rtype = 'over'
        print('samples <= features, resampling strategy replaced to over-sampling')

    # resampling algorithms
    if rtype == 'under':
        if (numeric is True) & (categorical is False):
            print('AllKNN is used for resampling')
            allknn = AllKNN(sampling_strategy='not minority')
            x, y = allknn.fit_resample(x, y)
        elif categorical is True:
            print('RandomUnderSampler is used for resampling')
            rs = RandomUnderSampler(random_state=rand, sampling_strategy='majority')
            x, y = rs.fit_resample(x, y)
        else:
            raise Exception("There is no categorical or numeric features")
    elif rtype == 'over':
        if (numeric is True) & (categorical is False):
            print('SMOTE is used for resampling')
            smote = SMOTE(sampling_strategy='not majority', random_state=rand)
            x, y = smote.fit_resample(x, y)
        elif (numeric is True) & (categorical is True):
            print('SMOTENC is used for resampling')
            smotenc = SMOTENC(categorical_features=cat_features, sampling_strategy='not majority', random_state=rand)
            x, y = smotenc.fit_resample(x, y)
        elif (numeric is False) & (categorical is True):
            print('SMOTEN is used for resampling')
            smoten = SMOTEN(sampling_strategy='not majority', random_state=rand)
            x, y = smoten.fit_resample(x, y)
        else:
            raise Exception("There is no categorical or numeric features")
    elif rtype == 'combined':
        if (numeric is True) & (categorical is False):
            print('SMOTEENN is used for resampling')
            smote_enn = SMOTEENN(random_state=rand)
            x, y = smote_enn.fit_resample(x, y)
        elif (numeric is True) & (categorical is True):
            print('SMOTENC + RandomUnderSampler are used')
            smotenc = SMOTENC(categorical_features=cat_features, sampling_strategy='not majority', random_state=rand)
            rs = RandomUnderSampler(random_state=rand, sampling_strategy='majority')
            x, y = smotenc.fit_resample(x, y)
            x, y = rs.fit_resample(x, y)
        elif (numeric is False) & (categorical is True):
            print('SMOTEN + RandomUnderSampler are used')
            smoten = SMOTEN(sampling_strategy='not majority', random_state=rand)
            rs = RandomUnderSampler(random_state=rand, sampling_strategy='majority')
            x, y = smoten.fit_resample(x, y)
            x, y = rs.fit_resample(x, y)
        else:
            raise Exception("There is no categorical or numeric features")
    elif rtype == 'auto':
        raise Exception("Not implemented")
    else:
        raise Exception("Wrong resampling type")

    dist = list(Counter(y).items())
    print('Class distribution after resampling:', dist)

    return x, y
