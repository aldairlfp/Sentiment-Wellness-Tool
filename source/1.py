import pandas as pd
from empath import Empath
import collections
import numpy as np
import os

raw_data = pd.read_csv('D:\School\ML\Sentiment-Wellness-Tool\data\data.csv')

bitmap_data = None

#region Data Transformation
def get_features(data):
    data_mod = data.dropna()

    lexicon = Empath()
    result = []
    labels = []

    for data in data_mod['statement']:
        result.append(lexicon.analyze(data))
    for label in data_mod['status']:
        labels.append(label)
    return result, labels


#endregion