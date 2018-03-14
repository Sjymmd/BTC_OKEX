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
from config import apikey,secretkey
warnings.filterwarnings("ignore")
#
okcoinRESTURL = 'www.okex.com'
apikey=apikey
secretkey=secretkey
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
okcoinfuture = OKCoinFuture(okcoinRESTURL, apikey, secretkey)
#
class Okex_Api:

    def __init__(self):
        self._Kline={'1min':'1min','3min':'3min','5min':'5min','15min':'15min','30min':'30min','1day':'1day','3day':'3day','1week':'1week','1hour':'1hour','2hour':'2hour','4hour':'4hour','6hour':'6hour','12hour':'12hour'}
        self._Lenth = 24*1000
        self._KlineChosen = '1hour'
        self._Watch_Coin = 'snt'
        while True:
            try:
                self._USDT_CNY = okcoinfuture.exchange_rate()['rate']
                break
            except:
                print('OKEX_Error')
                time.sleep(60)

        self._EndLenth = 0

    def Input(self):
        Str = '\n'.join(self._Kline.values())
        Input_Kline = input('输入时间区间,选择如下\n %s\n(default 1hour):'%Str)
        if Input_Kline:
            self._KlineChosen = self._Kline[Input_Kline]
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
        data = pd.DataFrame(okcoinSpot.getKline(self._Kline[self._KlineChosen], self._Lenth, self._EndLenth, Coin)).iloc[:, ]
        data = data.iloc[:-1,:]
        data[5] = data[5].apply(pd.to_numeric)
        if data.iloc[-1, 5] < 1000:
            # print('上一小时成交量小于1K不计数')
            return 0,0,0,0,0,0
        else:
            data = data[data[5]>=1000]
            data.reset_index(drop=True)
            Increase = (float(data.iloc[ -1, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
            Increase = str('%.2f'%(Increase)+'%')
            price = float(data.iloc[- 1, 4])
            Cny = round(price*self._USDT_CNY,2)
            Volume = data[5]
            # Volume = data.iloc[:, 5].apply(pd.to_numeric)
            Volume_Mean = round(Volume.mean()/1000,2)
            Volume_Pre  = round(Volume.iloc[-1]/1000,2)
            Volume_Pre_P = int(((Volume.iloc[-1]/Volume.iloc[-2])-1)*100)
            Volume_Inc = int(((Volume_Pre-Volume_Mean)/Volume_Mean)*100)
            return Cny,Increase,Volume_Mean,Volume_Pre,Volume_Pre_P,Volume_Inc

    def GetDataframe(self,DataFrame,Coin):
        Cny, Increase, Volume_Mean, Volume_Pre, Volume_Pre_P,Volume_Inc = self.GetKline(Coin)
        Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                                   'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P,'_VolumeM':Volume_Inc})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        return DataFrame

    def GetDataCoin(self,Coin,Clean = True):
        try:
            DataFrame = pd.DataFrame(columns=(
            "Coin", "Cny", "High", "Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
            data = pd.DataFrame(
                okcoinSpot.getKline(self._Kline[self._KlineChosen], self._Lenth, self._EndLenth,
                                    Coin)).iloc[:self._Lenth, ]
            data[5] = data.iloc[:, 5].apply(pd.to_numeric)
            if Clean:
                data = data[data[5] >= 1000]
                data = data.reset_index(drop=True)
            Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
            # Increase = str('%.2f' % (Increase) + '%')
            price = float(data.iloc[0, 4])
            Hi_price = round(float((data.iloc[0, 2])) * self._USDT_CNY, 2)
            Lo_price = round(float((data.iloc[0, 3])) * self._USDT_CNY, 2)
            Cny = round(price * self._USDT_CNY, 4)
            Volume = float(data.iloc[0, 5])
            Volume_Mean = round(Volume / 1000, 2)
            Volume_Pre = round(Volume / 1000, 2)
            Volume_Pre_P = 0
            if Volume_Mean == 0:
                Volume_Inc = 0
            else:
                Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
            Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'High': Hi_price, 'Low': Lo_price, 'Inc': Increase,
                                   'Volume_Pre_K': Volume_Pre,
                                   'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
            DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
            for lenth in range(1, len(data)-1):
                try:
                    Increase = (float(data.iloc[lenth, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
                    # Increase = str('%.2f' % (Increase) + '%')
                    price = float(data.iloc[lenth, 4])
                    Hi_price = round(float((data.iloc[lenth, 2])) * self._USDT_CNY, 2)
                    Lo_price = round(float((data.iloc[lenth, 3])) * self._USDT_CNY, 2)
                    Cny = round(price * self._USDT_CNY, 4)
                    Volume = data.iloc[:lenth + 1, 5].apply(pd.to_numeric)
                    Volume_Mean = round(Volume.mean() / 1000, 2)
                    Volume_Pre = round(Volume.iloc[lenth] / 1000, 2)
                    Volume_Pre_P = round((Volume[lenth] / Volume[lenth - 1]) - 1, 2)
                    Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
                    Timeshrft = pd.Series(
                        {'Coin': Coin, 'Cny': Cny, 'High': Hi_price, 'Low': Lo_price, 'Inc': Increase,
                         'Volume_Pre_K': Volume_Pre,
                         'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
                    DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
                except:
                    break
            return DataFrame
            # print(DataFrame)
        except:
            time.sleep(5)
            print('%s error' % Coin)

def Run(default = True):
        Main = Okex_Api()
        try:
            Coin = Main.GetCoin()
            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')
            print(now)
            StartTime = time.time()
        except:
            time.sleep(5)
            print('MainGetCoin_Error')
        if default :
            Main.Input()
        else:
            print('使用默认参数配置')
        DataFrame = pd.DataFrame(columns=("Coin", "Cny", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))

        for x in Coin[:int(Main._CoinLenth)]:
            try:
                DataFrame = Main.GetDataframe(DataFrame, x)
            except:
                # print('%s 读取失败' % x)
                continue
        DataFrame['Volume_Cny_K'] =DataFrame['Cny']*DataFrame['Mean_Volume_K']
        Mean_Mean_Volume_K = DataFrame['Volume_Cny_K'].mean()
        DataFrame = DataFrame[DataFrame.Volume_Cny_K >=Mean_Mean_Volume_K]
        DataFrame = DataFrame[DataFrame._VolumeS >1]
        DataFrame = DataFrame.sort_values(by='_VolumeS', ascending=False)
        DataFrame.pop('Volume_Cny_K')
        DataFrame = DataFrame.iloc[:10, ]
        Watch_Coin = str(Main._Watch_Coin + '_usdt')
        DataFrame = Main.GetDataframe(DataFrame, Watch_Coin)
        DataFrame =DataFrame.drop_duplicates(['Coin'])
        DataFrame = DataFrame.sort_values(by='_VolumeS', ascending=False)
        DataFrame=DataFrame.reset_index(drop=True)
        for x in (DataFrame.index):
            for columns in (-2,-1):
                DataFrame.iloc[x, columns] = str('%d' %DataFrame.iloc[x, columns] + '%')
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
    while True:
        sched.add_job(job,'cron', minute = 5)
        try:
            sched.start()
        except:
            print('定时任务出错')
            time.sleep(20)
            continue




