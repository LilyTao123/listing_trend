import pandas as pd
from trend import *

from sklearn.metrics import accuracy_score

args = {
    'lr_slope_threshold': 2,
    'trend_method': 'mix',
    'extreme_std': 3
}

analyzer = SalesTrendAnalyzer(**args)


def one_record_test(data):
    print(analyzer.analyze_trend(data))

data = [1,2,3,4,5,6]
print(one_record_test(data))