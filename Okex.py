#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding: utf-8
"""
Created on Sun Nov 24  2017

@author: Sjymmd

E-mail:1005965744@qq.com

BTC_Wallet:16ZA51dKeqQneyMSdhrZqugcsVTTgsEo7Y
"""
#
from OkcoinSpotAPI import *
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
#
okcoinRESTURL = 'www.okex.com'
apikey='1f74f0f9-54c2-4e1c-b653-3b3d2b2995d8'
secretkey='A04BBDEDC2B0B4436D853AA90BD4DD2B'
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
okcoinfuture = OKCoinFuture(okcoinRESTURL, apikey, secretkey)
#
Kline=['1min','3min','5min','15min','30min','1day','3day','1week','1hour','2hour','4hour','6hour','12hour']
Kline_Type = 5
lenth=1
global true
true = ''
global false
false = ''
while True:
    try:
        CoinType=eval(okcoinSpot.userinfo())['info']['funds']['free']
        break
    except:
        print('')
        continue
Coin = []
for (key,value) in CoinType.items():
    key = str(key+'_usdt')
    Coin.append(key)
# print(okcoinSpot.userinfo())
DataFrame = pd.DataFrame(columns=("Coin","CNY","Inc"))
for x in Coin:
    try:
        Inc = pd.DataFrame(okcoinSpot.getKline(Kline[Kline_Type], lenth, '0',x)).iloc[:,]
        increase = (float(Inc.iloc[lenth-1,4])-float(Inc.iloc[0,1]))/float(Inc.iloc[0,1])*100
        # increase = str('%d'%(increase)+'%')
        price = float(Inc.iloc[lenth-1,4])
        USD_CNY = okcoinfuture.exchange_rate()['rate']
        Timeshrft = pd.Series({'Coin':x,'Inc':increase,'CNY':price*USD_CNY})
        DataFrame = DataFrame.append(Timeshrft,ignore_index=True)
    except:
        continue
DataFrame = DataFrame.sort_values(by='Inc',ascending=False)
DataFrame_Chart = DataFrame
for x in DataFrame.index:
    DataFrame.iloc[x,2] = str('%d'%DataFrame.iloc[x,2]+'%')
print('Inc'+'%s' %Kline[Kline_Type])
print(DataFrame)
lenth = np.arange(DataFrame_Chart.shape[0])
plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(lenth,DataFrame_Chart['Inc'], 'g-', linewidth=1.0)
plt.show()

# print(timeStamp)
# func = lambda x:time.localtime(x)
# funcn = lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x))
# T = timeStamp.apply(func)

# print(pd.DataFrame(okcoinSpot.getKline('1day', '1', '0','snt_btc')))
# print (u' 现货行情 ')
# print (okcoinSpot.ticker('snt_btc'))
# print (u' 现货深度 ')
# type=okcoinSpot.depth('btc_usdt')
# a=[]
# print(type)
# for n in range(2):
#     for x in type.keys():
#         for y in range(len(type[x])):
#             a.append(type[x][1][n])
#         print(len(type[x]))
#         print(x)







# print(pd.DataFrame(a))
#print (u' 现货历史交易信息 ')
#print (okcoinSpot.trades())

#print (u' 用户现货账户信息 ')
#print (okcoinSpot.userinfo())

#print (u' 现货下单 ')
# print (okcoinSpot.trade('ltc_usdt','buy','0.1','0.2'))

#print (u' 现货批量下单 ')
#print (okcoinSpot.batchTrade('ltc_usd','buy','[{price:0.1,amount:0.2},{price:0.1,amount:0.2}]'))

#print (u' 现货取消订单 ')
#print (okcoinSpot.cancelOrder('ltc_usd','18243073'))

#print (u' 现货订单信息查询 ')
#print (okcoinSpot.orderinfo('ltc_usd','18243644'))

#print (u' 现货批量订单信息查询 ')
#print (okcoinSpot.ordersinfo('ltc_usd','18243800,18243801,18243644','0'))

#print (u' 现货历史订单信息查询 ')
#print (okcoinSpot.orderHistory('ltc_usd','0','1','2'))

#print (u' 期货行情信息')
#print (okcoinFuture.future_ticker('ltc_usd','this_week'))

#print (u' 期货市场深度信息')
#print (okcoinFuture.future_depth('btc_usd','this_week','6'))

#print (u'期货交易记录信息')
#print (okcoinFuture.future_trades('ltc_usd','this_week'))

#print (u'期货指数信息')
#print (okcoinFuture.future_index('ltc_usd'))

#print (u'美元人民币汇率')
#print (okcoinFuture.exchange_rate())

#print (u'获取预估交割价')
#print (okcoinFuture.future_estimated_price('ltc_usd'))

#print (u'获取全仓账户信息')
#print (okcoinFuture.future_userinfo())

#print (u'获取全仓持仓信息')
#print (okcoinFuture.future_position('ltc_usd','this_week'))

#print (u'期货下单')
#print (okcoinFuture.future_trade('ltc_usd','this_week','0.1','1','1','0','20'))

#print (u'期货批量下单')
#print (okcoinFuture.future_batchTrade('ltc_usd','this_week','[{price:0.1,amount:1,type:1,match_price:0},{price:0.1,amount:3,type:1,match_price:0}]','20'))

#print (u'期货取消订单')
#print (okcoinFuture.future_cancel('ltc_usd','this_week','47231499'))

#print (u'期货获取订单信息')
#print (okcoinFuture.future_orderinfo('ltc_usd','this_week','47231812','0','1','2'))

#print (u'期货逐仓账户信息')
#print (okcoinFuture.future_userinfo_4fix())

#print (u'期货逐仓持仓信息')
#print (okcoinFuture.future_position_4fix('ltc_usd','this_week',1))