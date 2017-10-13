import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# import seaborn as sns
from Mysql import *
from math import log, sqrt, exp
import scipy
from strategy import *
import mpl_toolkits.mplot3d

class analyse:
    def cut(deep):
        deep['bid_price'] = ''
        deep['bid_volume'] = ''
        deep['ask_price'] = ''
        deep['ask_price'] = ''
        for i in range(len(deep)):
            deep.ix[i, 'bid_price'] = deep.ix[i, 'bids'][0]
            deep.ix[i, 'bid_volume'] = deep.ix[i, 'bids'][1]
            deep.ix[i, 'ask_price'] = deep.ix[i, 'asks'][0]
            deep.ix[i, 'ask_volume'] = deep.ix[i, 'asks'][1]
        del deep['asks']
        del deep['bids']
        deep['bid_price'] = deep['bid_price'].astype('float64')
        deep['bid_volume'] = deep['bid_volume'].astype('float64')
        deep['ask_price'] = deep['ask_price'].astype('float64')
        deep['ask_price'] = deep['ask_price'].astype('float64')
        return deep

    def bid_ask_vol_diff(deep):
        bidvol10 = deep['bid_volume'][:10]
        askvol10 = deep['ask_volume'][-10:]
        diff = bidvol10.sum() - askvol10.sum()
        return diff  # diff>0是入场条件1

    def bid_ask_price_diff(deep):
        bidprice10 = deep['bid_price'][:10]
        askprice10 = deep['ask_price'][-10:]
        bid_diff = bidprice10.max() - bidprice10.min()
        ask_diff = askprice10.max() - askprice10.min()
        diff = bid_diff - ask_diff  # 小于0是入场条件
        return diff

    def bid_ask_bigvol(deep):
        bidvol10 = deep['bid_volume'][:10]
        askvol10 = deep['ask_volume'][-10:]
        diff = bidvol10.max() - askvol10.max()  # 大于0是入场条件
        return diff

    def linear_model_main(X_parameters, Y_parameters, predict_value):
        regr = linear_model.LinearRegression()
        regr.fit(X_parameters, Y_parameters)
        predict_outcome = regr.predict(predict_value)
        predictions = {}
        predictions['intercept'] = regr.intercept_
        predictions['coefficient'] = regr.coef_
        predictions['predicted_value'] = predict_outcome
        return predictions

