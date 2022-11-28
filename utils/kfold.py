import numpy as np
import pandas as pd

class KFold(object):
    """
    KFold: Group split by group_col or random_split
    """

    def __init__(self, random_seed, k_folds=10, flag_name='fold_flag'):
        self.k_folds = k_folds
        self.flag_name = flag_name
        np.random.seed(random_seed)

    def group_split(self, train_df, group_col):
        group_value = list(set(train_df[group_col]))
        group_value.sort()
        fold_flag = [i % self.k_folds for i in range(len(group_value))]
        np.random.shuffle(fold_flag)
        train_df = train_df.merge(pd.DataFrame({group_col: group_value, self.flag_name: fold_flag}), how='left',
                                  on=group_col)
        return train_df

    def random_split(self, train_df):
        fold_flag = [i % self.k_folds for i in range(len(train_df))]
        np.random.shuffle(fold_flag)
        train_df[self.flag_name] = fold_flag
        return train_df

    def stratified_split(self, train_df, group_col):
        train_df[self.flag_name] = 1
        train_df[self.flag_name] = train_df.groupby(by=[group_col])[self.flag_name].rank(ascending=True,
                                                                                         method='first').astype(int)
        train_df[self.flag_name] = train_df[self.flag_name].sample(frac=1.0).reset_index(drop=True)
        train_df[self.flag_name] = train_df[self.flag_name] % self.k_folds
        return train_df