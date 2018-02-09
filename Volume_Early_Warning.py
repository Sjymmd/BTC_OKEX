#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding: utf-8
"""
Created on Thurs Feb 8  2018

@author: Sjymmd
E-mail:1005965744@qq.com
"""
#
from OkcoinSpotAPI import *
import pandas as pd
import time
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#
okcoinRESTURL = 'www.okex.com'
apikey='1f74f0f9-54c2-4e1c-b653-3b3d2b2995d8'
secretkey='A04BBDEDC2B0B4436D853AA90BD4DD2B'
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
okcoinfuture = OKCoinFuture(okcoinRESTURL, apikey, secretkey)
#
class Okex_Api:

    def __init__(self):
        self._Kline={'1min':'1min','3min':'3min','5min':'5min','15min':'15min','30min':'30min','1day':'1day','3day':'3day','1week':'1week','1hour':'1hour','2hour':'2hour','4hour':'4hour','6hour':'6hour','12hour':'12hour'}
        self._Lenth = 24
        self._KlineChose = '1hour'
        self._Watch_Coin = 'snt'
        self._USD_CNY = okcoinfuture.exchange_rate()['rate']

    def Input(self):
        Str = '\n'.join(self._Kline.values())
        Input_Kline = input('输入时间区间,选择如下\n %s\n(default 1hour):'%Str)
        if Input_Kline:
            self._KlineChose = self._Kline[Input_Kline]
        Input_Num = input('输入数量(default 24):')
        if Input_Num:
            self._Lenth = Input_Num
        Input_Coin_Num = input('输入币循环数量(default %s):'%self._CoinLenth)
        if Input_Coin_Num:
            self._CoinLenth = Input_Coin_Num
        Input_Watch_Coin = input('输入紧盯币(default %s):'%self._Watch_Coin)
        if Input_Watch_Coin:
            self._Watch_Coin = Input_Watch_Coin

    def GetCoin(self):
        global true
        true = ''
        global false
        false = ''
        while True:
            try:
                CoinType = eval(okcoinSpot.userinfo())['info']['funds']['free']
                break
            except:
                print('GetCoin_Error')
                continue
        Coin = []
        for (key, value) in CoinType.items():
            key = str(key + '_usdt')
            Coin.append(key)
        self._CoinLenth = len(Coin)
        return Coin

    def GetKline(self,Coin):
        data = pd.DataFrame(okcoinSpot.getKline(self._Kline[self._KlineChose], self._Lenth, '0', Coin)).iloc[:, ]
        Increase = (float(data.iloc[self._Lenth - 1, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
        Increase = str('%d'%(Increase)+'%')
        price = float(data.iloc[self._Lenth - 1, 4])
        Cny = price*self._USD_CNY
        Volume = data.iloc[:, 5].apply(pd.to_numeric)
        Volume_Mean = Volume.mean()
        Volume_Pre  = Volume[self._Lenth-2]
        Volume_Pre_P = round((Volume[self._Lenth-2]/Volume[self._Lenth-3]),2)
        return Cny,Increase,Volume_Mean/1000,Volume_Pre/1000,Volume_Pre_P

    def GetDataframe(self,DataFrame,Coin):
        Cny, Increase, Volume_Mean, Volume_Pre, Volume_Pre_P = self.GetKline(Coin)
        Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                                   'Mean_Volume_K': Volume_Mean, '_Volume': Volume_Pre_P})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        return DataFrame

def Run(default = True):
        Main = Okex_Api()
        Coin = Main.GetCoin()
        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now)
        StartTime = time.time()
        if default :
            Main.Input()
        else:
            print('使用默认参数配置')
        DataFrame = pd.DataFrame(columns=("Coin", "Cny", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_Volume"))

        for x in Coin[:int(Main._CoinLenth)]:
            try:
                DataFrame = Main.GetDataframe(DataFrame, x)
            except:
                # print('%s 读取失败' % x)
                continue
        DataFrame['Temp']=(DataFrame['Volume_Pre_K']-DataFrame['Mean_Volume_K'])/DataFrame['Mean_Volume_K']
        DataFrame['Temp_2'] =DataFrame['Cny']*DataFrame['Mean_Volume_K']
        Mean_Mean_Volume_K = DataFrame['Temp_2'].mean()
        DataFrame = DataFrame[DataFrame.Temp_2>=Mean_Mean_Volume_K]
        DataFrame = DataFrame[DataFrame._Volume >1]
        DataFrame = DataFrame.sort_values(by='Temp', ascending=False)
        DataFrame.pop('Temp')
        DataFrame.pop('Temp_2')
        DataFrame = DataFrame.iloc[:10, ]
        Watch_Coin = str(Main._Watch_Coin + '_usdt')
        DataFrame = Main.GetDataframe(DataFrame, Watch_Coin)
        DataFrame =DataFrame.drop_duplicates(['Coin'])
        if DataFrame.empty:
            print('没有符合的币种')
        else:
            print(DataFrame)
        EndTime = time.time()
        print('Using_Time: %d sec'%int(EndTime - StartTime))
if __name__=='__main__':

    def job():
        Run(False)
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler()
    sched.add_job(job,'cron', minute = 5)
    sched.start()
