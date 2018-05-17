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
import warnings
import numpy as np
from config import apikey, secretkey

warnings.filterwarnings("ignore")
#
okcoinRESTURL = 'www.okex.com'
apikey = apikey
secretkey = secretkey
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
okcoinfuture = OKCoinFuture(okcoinRESTURL, apikey, secretkey)


#
class Okex_Api:

    def __init__(self):
        self._Kline = {'1min': '1min', '3min': '3min', '5min': '5min', '15min': '15min', '30min': '30min',
                       '1day': '1day', '3day': '3day', '1week': '1week', '1hour': '1hour', '2hour': '2hour',
                       '4hour': '4hour', '6hour': '6hour', '12hour': '12hour'}
        self._Lenth = 61
        self._KlineChosen = '1hour'
        self._Watch_Coin = 'snt'
        while True:
            try:
                self._USDT_CNY = okcoinfuture.exchange_rate()['rate']
                break
            except:
                print('Get_USDT_Error~6.3')
                self._USDT_CNY = 6.3
                break
                # time.sleep(60)

        self._EndLenth = 0

    def Input(self):
        Str = '\n'.join(self._Kline.values())
        Input_Kline = input('输入时间区间,选择如下\n %s\n(default 1hour):' % Str)
        if Input_Kline:
            self._KlineChosen = self._Kline[Input_Kline]
        Input_Num = input('输入数量(default 24):')
        if Input_Num:
            self._Lenth = Input_Num
        Input_Coin_Num = input('输入币循环数量(default %s):' % self._CoinLenth)
        if Input_Coin_Num:
            self._CoinLenth = Input_Coin_Num
        Input_Watch_Coin = input('输入紧盯币(default %s):' % self._Watch_Coin)
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

    def GetKline(self, Coin):
        data_init = pd.DataFrame(
            okcoinSpot.getKline(self._Kline[self._KlineChosen], self._Lenth, self._EndLenth, Coin)).iloc[:, ]
        data = data_init.iloc[-25:-1,:]
        data[5] = data[5].apply(pd.to_numeric)
        if data.iloc[-1, 5] < 1000:
            # print('上一小时成交量小于1K不计数')
            return 0, 0, 0, 0, 0, 0
        else:

            MA60 = round(float(data_init.iloc[:-1, 4].apply(pd.to_numeric).mean() * self._USDT_CNY), 2) #60小时平均价格
            data = data[data[5] >= 1000]
            data.reset_index(drop=True)
            Increase = (float(data.iloc[-1, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
            Increase = str('%.2f' % (Increase) + '%')
            price = float(data.iloc[- 1, 4])
            Cny = round(price * self._USDT_CNY, 2)
            Volume = data[5]
            # Volume = data.iloc[:, 5].apply(pd.to_numeric)
            Volume_Mean = round(Volume.mean() / 1000, 2)
            Volume_Pre = round(Volume.iloc[-1] / 1000, 2)
            Volume_Pre_P = round(float(Volume.iloc[-1] / Volume.iloc[-2]),2 )#上一小时成交量涨幅比值
            Volume_Inc = int(((Volume_Pre - Volume_Mean) / Volume_Mean) * 100)  #平均成交量涨幅
            Volume_4 = round(float(Volume.iloc[-4:].mean()/Volume.iloc[-8:-4].mean()) ,2)#4小时平均成交量涨幅比值

            Data_Day = pd.DataFrame(
                okcoinSpot.getKline(self._Kline['1day'], 5, self._EndLenth, Coin)).iloc[:, ]
            MA_Day5 =round(float(Data_Day.iloc[:,4].apply(pd.to_numeric).mean()* self._USDT_CNY),2) #5日平均价格

            return Cny, Increase, Volume_Mean, Volume_Pre, Volume_Pre_P, Volume_Inc,Volume_4,MA60,MA_Day5

    def GetDataframe(self, DataFrame, Coin):
        Cny, Increase, Volume_Mean, Volume_Pre, Volume_Pre_P, Volume_Inc,Volume_4,MA60,MA_Day5 = self.GetKline(Coin)
        Timeshrft = pd.Series({'Coin': Coin, 'Price': Cny, 'Inc': Increase, 'V/K': Volume_Pre,
                               'MV/K': Volume_Mean, 'V_S': Volume_Pre_P, 'V_L': Volume_Inc,'V_4h':Volume_4,'MA60':MA60,'MA_Day5':MA_Day5})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        return DataFrame



def Run(default=True):
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
    if default:
        Main.Input()
    else:
        print('使用默认参数配置')
    DataFrame = pd.DataFrame(columns=('Coin','Price','Inc','V/K',
                               'MV/K', 'V_S', 'V_L','V_4h','MA60','MA_Day5'))

    for x in Coin[:int(Main._CoinLenth)]:
        try:
            DataFrame = Main.GetDataframe(DataFrame, x)
        except:
            # print('%s 读取失败' % x)
            continue
    DataFrame['Volume_Cny_K'] = DataFrame['Price'] * DataFrame['MV/K']
    Mean_Mean_Volume_K = DataFrame['Volume_Cny_K'].mean()
    DataFrame = DataFrame[DataFrame.Volume_Cny_K >= Mean_Mean_Volume_K]
    DataFrame = DataFrame[DataFrame.V_S > 1]
    DataFrame = DataFrame.sort_values(by='V_S', ascending=False)
    DataFrame.pop('Volume_Cny_K')
    DataFrame = DataFrame.iloc[:10, ]
    Watch_Coin = str(Main._Watch_Coin + '_usdt')
    DataFrame = Main.GetDataframe(DataFrame, Watch_Coin)
    DataFrame = DataFrame.drop_duplicates(['Coin'])
    DataFrame = DataFrame.sort_values(by='V_S', ascending=False)
    DataFrame = DataFrame.reset_index(drop=True)
    # for x in (DataFrame.index):
    #     for columns in (-2, -1):
    #         DataFrame.iloc[x, columns] = str('%d' % DataFrame.iloc[x, columns] + '%')
    for x in (DataFrame.index):
        DataFrame.iloc[x, -4] = str('%d' % DataFrame.iloc[x, -4] + '%')

    Wechat_Push = [Wechat.from_group,Wechat.friend]

    for name in Wechat_Push:

        if DataFrame.empty:
            # print('没有符合的币种')
            wechatmsg = '没有符合的币种'
            Wechat.msg(wechatmsg, name)
        else:
            # print(DataFrame)
            now = datetime.datetime.now()
            # now = now.strftime('%Y-%m-%d %H:%M:%S')
            for x in range(0, 2):
                wechatmsg = DataFrame.iloc[:, 3 * x:(x + 1) * 3].to_string()
                Wechat.msg(wechatmsg, name)
            wechatmsg = DataFrame.iloc[:, -4:].to_string()
            Wechat.msg(wechatmsg, name)
        time.sleep(5)

    print(DataFrame if not DataFrame.empty else '没有符合的币种')
    EndTime = time.time()
    print('Using_Time: %d sec' % int(EndTime - StartTime))


if __name__ == '__main__':

    from Class_Wechat import Wechat
    Wechat = Wechat()
    Wechat.Get_Chatrooms('PythonGroup')
    Wechat.Get_Friends('belief.')
    def job():
        Run(False)


    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler()
    while True:
        sched.add_job(job, 'cron', minute=5)
        # sched.add_job(job, 'interval', seconds=30)
        try:
            sched.start()
        except:
            print('定时任务出错')
            time.sleep(20)
            continue




