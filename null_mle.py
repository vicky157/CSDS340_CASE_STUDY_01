import pandas as pd
import numpy as np
from scipy.stats import norm

features = ["f" + str(i) for i in range(1, 31)]
features.append("class")
data = pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain.csv', names=features)
def replace_with_mle(column):
    if column.name != "class":
        non_missing_values = column[column != -1]
        mle_mean, mle_std = norm.fit(non_missing_values)
        column[column == -1] = np.random.normal(mle_mean, mle_std, sum(column == -1))
data.apply(replace_with_mle)
data.to_csv('/home/vicky/Desktop/csds340/case_study/spamTrain_mle.csv', index=False)
