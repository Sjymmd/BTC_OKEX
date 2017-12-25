# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:36:19 2017

@author: mjc

E-mail:1005965744@qq.com

Differentiate yourself in the world from anyone else.
"""
from OkcoinSpotAPI import *
from Mysql import *
import pandas as pd
from Analyse import *
import numpy as np
import datetime
import time
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# import seaborn as sns
from Mysql import *
from math import log, sqrt, exp
import scipy
from strategy import *
import mpl_toolkits.mplot3d
#####################################################初始数据
okcoinRESTURL = 'www.okex.com'
apikey='1f74f0f9-54c2-4e1c-b653-3b3d2b2995d8'
secretkey='A04BBDEDC2B0B4436D853AA90BD4DD2B'
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
#####################################################strategy
class strategy:
    def __init__(self, url, apikey, secretkey):
        self.__url = url
        self.__apikey = apikey
        self.__secretkey = secretkey

    def bbands_MA(M, N,ltc_btc):
        m = str(M)
        n = str(N)
        try:
            kline = pd.DataFrame(okcoinSpot.getKline('1min', m, '0',ltc_btc))
        except ValueError as e:
            print('bbands_MA Error')
        mid = kline.iloc[::, 4].mean()
        std = kline.iloc[::, 4].std()
        upper = mid + N * std
        lower = mid - N * std
        # ma = kline.iloc[::, 4].mean()
        result = [upper, mid, lower]
        return result

    def BUY_T(info_usdt, price_buy, amount, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation):
        global true
        true = ''
        global false
        false = ''
        if info_usdt >= price_buy * amount and amount>=0.1:
            price_buy=price_buy+1
            buy = okcoinSpot.trade(ltc_btc, 'buy', price_buy, amount)
            try:
                buyid = str(eval(buy)['order_id'])
            except:
                print('BuyId Error')
            # time.sleep(5)
            OKCoinSpot.progresive(6)
            cancel_buy = okcoinSpot.cancelOrder(ltc_btc, buyid)
            if "error_code" in cancel_buy:
                true = ''
                false = ''
                info = initial.info_get(1)
                (info_ltc, info_total, info_usdt, info_eth) = (info[0], info[1], info[2], info[3])
                mysql.COST_CALCUATION(price_buy, 0.998 * amount, 0,info_ltc,info_eth,info_usdt,db,Annotation)
                print('交易单价', price_buy, '交易数量', amount)
                print('成交量差', analyse.bid_ask_vol_diff(deep), '成交价差', analyse.bid_ask_price_diff(deep))
                print('月初资本', InI_COST, '当前资本', NOW_COST, '静态收益', NOW_COST - InI_COST)
            else:
                print('取消订单号', cancel_buy)
        else:
            print("usdt不足")
            print("交易usdt数", amount)

    def BUY_T_ETH(info_usdt, price_buy, amount, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation):
        global true
        true = ''
        global false
        false = ''
        if info_usdt >= price_buy * amount and amount>=0.01:
            price_buy=price_buy+2
            buy = okcoinSpot.trade(ltc_btc, 'buy', price_buy, amount)
            try:
                buyid = str(eval(buy)['order_id'])
            except:
                print('BuyId Error')
            # time.sleep(5)
            OKCoinSpot.progresive(6)
            cancel_buy = okcoinSpot.cancelOrder(ltc_btc, buyid)
            if "error_code" in cancel_buy:
                true = ''
                false = ''
                info = initial.info_get(1)
                (info_ltc, info_total, info_usdt, info_eth) = (info[0], info[1], info[2], info[3])
                mysql.COST_CALCUATION_ETH(price_buy, 0.999 * amount, 0,info_ltc,info_eth,info_usdt,db,Annotation)
                print('交易单价', price_buy, '交易数量', amount)
                print('成交量差', analyse.bid_ask_vol_diff(deep), '成交价差', analyse.bid_ask_price_diff(deep))
                print('月初资本', InI_COST, '当前资本', NOW_COST, '静态收益', NOW_COST - InI_COST)
            else:
                print('取消订单号', cancel_buy)
        else:
            print("usdt不足")
            print("交易usdt数", amount)

    def SELL_T(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation):
        global true
        true = ''
        global false
        false = ''
        if info_ltc > amount_sell and info_ltc>=0.1:
            price_sell=price_sell-1
            sell = okcoinSpot.trade(ltc_btc, 'sell', price_sell, amount_sell)
            try:
                sellid = str(eval(sell)['order_id'])
            except:
                print('SellId Error')
            OKCoinSpot.progresive(6)
            cancel_sell = okcoinSpot.cancelOrder(ltc_btc, sellid)
            if "error_code" in cancel_sell:
                true = ''
                false = ''
                info = initial.info_get(1)
                (info_ltc, info_total, info_usdt, info_eth) = (info[0], info[1], info[2], info[3])
                charge_usdt_sell = price_sell * 0.002 * amount_sell
                benefit = (price_sell - AVG_COST - charge_usdt / 0.1) * amount_sell
                mysql.mysql_insert_SELL_OUT(charge_usdt_sell, benefit,db,Annotation)
                mysql.COST_CALCUATION_SELL(price_sell, amount_sell, charge_usdt_sell,info_ltc,info_eth,info_usdt,db,Annotation)
                # mysql.COST_CALCUATION_SELL(close, 0.1, charge_usdt)
                benefit_sum = mysql.SELL_OUT(30,db)[0]
                datetime_Start = mysql.SELL_OUT(30,db)[1]
                datetime_End = mysql.SELL_OUT(30,db)[2]
                print('卖出信号')
                print('月初资本', InI_COST, '当前资本', NOW_COST, '静态收益', NOW_COST - InI_COST)
                print('成交价', price_sell, '移动平均', AVG_COST, '交易数量', amount_sell, '卖出手续费', charge_usdt_sell, '\n', '收益',
                      benefit, '月累计收益', benefit_sum)
                print('成交量差', analyse.bid_ask_vol_diff(deep), '成交价差', analyse.bid_ask_price_diff(deep))
                print(datetime_Start, 'to', datetime_End)
        else:
            print('交易LTC数', amount_sell)
            print("LTC不足")

    def SELL_T_ETH(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation):
        global true
        true = ''
        global false
        false = ''
        if info_ltc > amount_sell and info_ltc>=0.01:
            price_sell=price_sell-1
            sell = okcoinSpot.trade(ltc_btc, 'sell', price_sell, amount_sell)
            try:
                sellid = str(eval(sell)['order_id'])
            except:
                print('SellId Error')
            OKCoinSpot.progresive(6)
            cancel_sell = okcoinSpot.cancelOrder(ltc_btc, sellid)
            if "error_code" in cancel_sell:
                true = ''
                false = ''
                info = initial.info_get(1)
                (info_ltc, info_total, info_usdt, info_eth) = (info[0], info[1], info[2], info[3])
                # info = initial.info_get(1)
                # (info_ltc, info_total, info_usdt, info_eth) = (info[0], info[1], info[2], info[3])
                charge_usdt_sell = price_sell * 0.001 * amount_sell
                benefit = (price_sell - AVG_COST - charge_usdt / 0.01) * amount_sell
                mysql.mysql_insert_SELL_OUT(charge_usdt_sell, benefit,db,Annotation)
                mysql.COST_CALCUATION_SELL_ETH(price_sell, amount_sell, charge_usdt_sell,info_ltc,info_eth,info_usdt,db,Annotation)
                # mysql.COST_CALCUATION_SELL(close, 0.1, charge_usdt)
                benefit_sum = mysql.SELL_OUT(30,db)[0]
                datetime_Start = mysql.SELL_OUT(30,db)[1]
                datetime_End = mysql.SELL_OUT(30,db)[2]
                print('卖出信号')
                print('月初资本', InI_COST, '当前资本', NOW_COST, '静态收益', NOW_COST - InI_COST)
                print('成交价', price_sell, '移动平均', AVG_COST, '交易数量', amount_sell, '卖出手续费', charge_usdt_sell, '\n', '收益',
                      benefit, '月累计收益', benefit_sum)
                print('成交量差', analyse.bid_ask_vol_diff(deep), '成交价差', analyse.bid_ask_price_diff(deep))
                print(datetime_Start, 'to', datetime_End)
        else:
            print('交易LTC数', amount_sell)
            print("LTC不足")

    def bid_ask_vol_diff(deep,db):
        if analyse.bid_ask_vol_diff(deep)<0:
            try:
                INSERT_TMP=float(analyse.bid_ask_vol_diff(deep))
                BID_AVG = Mysql_KPI.ask_vl_SELECT(1,db)
            except:
                print('INSERT ASK ERROR')
        else:
            try:
                INSERT_TMP = float(analyse.bid_ask_vol_diff(deep))
                BID_AVG = Mysql_KPI.bid_vl_SELECT(1,db)
            except:
                print('INSERT BID ERROR')
        return BID_AVG,INSERT_TMP

    def bid_ask_vol_diff_eth(deep,db):
        if analyse.bid_ask_vol_diff(deep)<0:
            try:
                INSERT_TMP=float(analyse.bid_ask_vol_diff(deep))
                BID_AVG = Mysql_KPI.ask_vl_SELECT_eth(1,db)
            except:
                print('INSERT ASK ERROR')
        else:
            try:
                INSERT_TMP = float(analyse.bid_ask_vol_diff(deep))
                BID_AVG = Mysql_KPI.bid_vl_SELECT_eth(1,db)
            except:
                print('INSERT BID ERROR')
        return BID_AVG,INSERT_TMP

    def amount_buy(close,db):
        if mysql.INI_COST(1,db)[1] > 0:
            amount_buy = (mysql.mysql_SELECT_BUY_SELL(1,db)[2] * (
            1 - ((close - mysql.INI_COST(1,db)[1]) / mysql.INI_COST(1,db)[1]))) / close
            if amount_buy >0.1:
                amount_buy = float('%.2f' % amount_buy)
            else:
                amount_buy=0.1
        else:
            amount_buy = 0.1

        return amount_buy

    def amount_buy_eth(close,db):
        if mysql.INI_COST(1,db)[2] > 0:
            amount_buy = (mysql.mysql_SELECT_BUY_SELL(1,db)[2] * (
            1 - ((close - mysql.INI_COST(1,db)[2]) / mysql.INI_COST(1,db)[2]))) / close

            if amount_buy>0.01:
                amount_buy = float('%.3f' % amount_buy)
            else:
                amount_buy=0.01
        else:
            amount_buy = 0.01

        return amount_buy


    def amount_sell(close,db):
        if mysql.INI_COST(1,db)[1] > 0:
            amount_sell_TMP = (
            mysql.mysql_SELECT_BUY_SELL(1,db)[1] * ((close - mysql.INI_COST(1,db)[1]) / mysql.INI_COST(1,db)[1]))
            if amount_sell_TMP > 0.1:
                amount_sell = float('%.2f' % amount_sell_TMP)
            else:
                amount_sell=0.1
        else:
            amount_sell = 0.1

        return amount_sell

    def amount_sell_eth(close,db):
        if mysql.INI_COST(1,db)[2] > 0:
            amount_sell_TMP = (
            mysql.mysql_SELECT_BUY_SELL(1,db)[3] * ((close - mysql.INI_COST(1,db)[2]) / mysql.INI_COST(1,db)[2]))
            if amount_sell_TMP > 0.01:
                amount_sell = float('%.3f' % amount_sell_TMP)
            else:
                amount_sell = 0.01
        else:
            amount_sell = 0.01

        return amount_sell

    def amount_buy_HV(info_usdt,price_buy,InI_COST,NOW_COST,deep,ltc_btc,db,half,Annotation):
        amount_buy_cal = info_usdt*half / price_buy
        if amount_buy_cal>0.1:
            amount_buy = math.floor(amount_buy_cal * 100) / 100
            amount = float('%.2f' % amount_buy)
        else:
            amount=0.1
        strategy.BUY_T(info_usdt, price_buy, amount, InI_COST, NOW_COST, deep,ltc_btc,db,Annotation)


    def amount_buy_HV_ETH(info_usdt,price_buy,InI_COST,NOW_COST,deep,ltc_btc,db,half,Annotation):
        amount_buy_cal = info_usdt*half / price_buy
        if amount_buy_cal>0.01:
            amount_buy = math.floor(amount_buy_cal * 1000) / 1000
            amount = float('%.3f' % amount_buy)
        else:
            amount=0.01
        strategy.BUY_T_ETH(info_usdt, price_buy, amount, InI_COST, NOW_COST, deep,ltc_btc,db,Annotation)


    def amount_sell_HV(info_ltc,price_sell,AVG_COST,charge_usdt,InI_COST,NOW_COST,deep,ltc_btc,db,half,Annotation):
        a=info_ltc*half
        if a>0.1:
            amount_sell_TMP = math.floor(info_ltc*half * 100) / 100
            amount_sell = float('%.2f' % amount_sell_TMP)
        else:
            amount_sell=0.1
        strategy.SELL_T(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation)

    def amount_sell_HV_ETH(info_ltc,price_sell,AVG_COST,charge_usdt,InI_COST,NOW_COST,deep,ltc_btc,db,half,Annotation):
        a=info_ltc*half
        if a>0.01:
            amount_sell_TMP = math.floor(info_ltc*half * 1000) / 1000
            amount_sell = float('%.3f' % amount_sell_TMP)
        else:
            amount_sell=0.01
        strategy.SELL_T_ETH(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,Annotation)

    def boll_buy(close,upper,ref_close,ref_upper,info_usdt,price_buy,InI_COST,NOW_COST,deep,ltc_btc,db):
        if close > upper and ref_close < ref_upper:
            print('布林线买入信号')
            amount = strategy.amount_buy(close,db)
            strategy.BUY_T(info_usdt, price_buy, amount, InI_COST, NOW_COST, deep,ltc_btc,db,'布林线买入信号')
        else:
            print('BOLL_BUY_MISS')

    def boll_buy_eth(close,upper,ref_close,ref_upper,info_usdt,price_buy,InI_COST,NOW_COST,deep,ltc_btc,db):
        if close > upper and ref_close < ref_upper:
            print('布林线买入信号')
            amount = strategy.amount_buy_eth(close,db)
            strategy.BUY_T_ETH(info_usdt, price_buy, amount, InI_COST, NOW_COST, deep,ltc_btc,db,'布林线买入信号')
        else:
            print('BOLL_BUY_MISS')

    def boll_sell(close,ref_close,ref_ma34,last,AVG_COST,charge_usdt,ma34,price_sell,info_ltc,InI_COST,NOW_COST,deep,ltc_btc,db):
        if close < ma34 and ref_close > ref_ma34 and last > AVG_COST + charge_usdt / 0.1 + 7:
            print('布林线卖出信号')
            amount_sell =strategy.amount_sell(close,db)
            strategy.SELL_T(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,'布林线卖出信号')
        else:
            print('BOLL_SELL_MISS')

    def boll_sell_eth(close,ref_close,ref_ma34,last,AVG_COST,charge_usdt,ma34,price_sell,info_ltc,InI_COST,NOW_COST,deep,ltc_btc,db):
        if close < ma34 and ref_close > ref_ma34 and last > AVG_COST + charge_usdt / 0.1 + 70:
            print('布林线卖出信号')
            amount_sell =strategy.amount_sell_eth(close,db)
            strategy.SELL_T_ETH(info_ltc, price_sell, amount_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST,deep,ltc_btc,db,'布林线卖出信号')
        else:
            print('BOLL_SELL_MISS')

class initial:

    def info_get(i):
        try:
            global true
            true = ''
            global false
            false = ''
            info_ltc = eval(eval(okcoinSpot.userinfo())['info']['funds']['free']['ltc'])
            info_eth = eval(eval(okcoinSpot.userinfo())['info']['funds']['free']['eth'])
            info_total = eval(eval(okcoinSpot.userinfo())['info']['funds']['asset']['total'])
            info_usdt = eval(eval(okcoinSpot.userinfo())['info']['funds']['free']['usdt'])
            info_freeze_usdt = eval(eval(okcoinSpot.userinfo())['info']['funds']['freezed']['usdt'])
            info_freeze_ltc = eval(eval(okcoinSpot.userinfo())['info']['funds']['freezed']['ltc'])

        except:
            print('info_get Error')
        return info_ltc,info_total,info_usdt,info_eth,info_freeze_usdt,info_freeze_ltc

    def boll_get(i,ltc_btc):
        try:
            ref_boll = strategy.bbands_MA(94, 2,ltc_btc)
            ref_upper = ref_boll[0]
            ref_lower = ref_boll[2]
            ref_close = okcoinSpot.getKline('1min', '1', '0',ltc_btc)[0][4]
            ref_ma34 = pd.DataFrame(okcoinSpot.getKline('1min', '34', '0',ltc_btc)).iloc[::, 4].mean()
        except:
            print('boll_get Error')
        return ref_upper,ref_lower,ref_close,ref_ma34

class PRE:

    def DATA(i,db,ltc_btc,bid_ask_vl):
        VOL_TMP = okcoinSpot.ticker(ltc_btc)
        volume = float(VOL_TMP["ticker"]["vol"])
        deep = pd.DataFrame(okcoinSpot.depth(ltc_btc))
        deep = analyse.cut(deep)
        high = float(initial.boll_get(1,ltc_btc)[0])
        low = float(initial.boll_get(1,ltc_btc)[1])
        last = float(initial.boll_get(1,ltc_btc)[2])
        # bid_ask_vl=float(analyse.bid_ask_vol_diff(deep))
        bid_ask_vl = bid_ask_vl
        RE_EMA_12 = Mysql_KPI.EMA_SELECT(1,db)[0]
        RE_EMA_26 = Mysql_KPI.EMA_SELECT(1,db)[1]
        RE_DEA = Mysql_KPI.EMA_SELECT(1,db)[2]
        EMA_12 = RE_EMA_12 * (11 / 13) + last * 2 / 13
        EMA_26 = RE_EMA_26 * 25 / 27 + last * 2 / 27
        DIF = EMA_12 - EMA_26
        DEA = RE_DEA * 8 / 10 + DIF * 2 / 10
        BAR = 2 * (DIF - DEA)
        Mysql_KPI.EMA_INSERT(EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume,db)
        return EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume

    def DATA_eth(i,db,ltc_btc,bid_ask_vl):
        VOL_TMP = okcoinSpot.ticker(ltc_btc)
        volume = float(VOL_TMP["ticker"]["vol"])
        deep = pd.DataFrame(okcoinSpot.depth(ltc_btc))
        deep = analyse.cut(deep)
        high = float(initial.boll_get(1,ltc_btc)[0])
        low = float(initial.boll_get(1,ltc_btc)[1])
        last = float(initial.boll_get(1,ltc_btc)[2])
        bid_ask_vl=bid_ask_vl
        RE_EMA_12 = Mysql_KPI.EMA_SELECT_ETH(1,db)[0]
        RE_EMA_26 = Mysql_KPI.EMA_SELECT_ETH(1,db)[1]
        RE_DEA = Mysql_KPI.EMA_SELECT_ETH(1,db)[2]
        EMA_12 = RE_EMA_12 * (11 / 13) + last * 2 / 13
        EMA_26 = RE_EMA_26 * 25 / 27 + last * 2 / 27
        DIF = EMA_12 - EMA_26
        DEA = RE_DEA * 8 / 10 + DIF * 2 / 10
        BAR = 2 * (DIF - DEA)
        Mysql_KPI.EMA_INSERT_ETH(EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume,db)
        return EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume


    def resistance(i,db):
        df_index = Mysql_KPI.DATA_Analyse(1,db)
        df_index['ii'] = range(len(df_index))
        t_df = df_index
        window_df = df_index.iloc[0:-1]
        window_df['dis_weight'] = window_df['ii'].map(lambda x: math.log(x + 2) / math.log(max(df_index['ii']) + 1))
        window_df['pri_weight'] = window_df['last'].map(
            lambda x: math.log(t_df.iloc[-1]['last'] / abs(x - t_df.iloc[-1]['last'])))
        numerator = window_df[window_df['last'] > t_df.iloc[-1]['last']][
            ['dis_weight', 'pri_weight', 'volume']].prod(axis=1).sum()
        denominator = window_df[['dis_weight', 'pri_weight', 'volume']].prod(axis=1).sum()
        resistance = numerator / denominator
        return resistance

    def resistance_eth(i,db):
        df_index = Mysql_KPI.DATA_Analyse_eth(1,db)
        df_index['ii'] = range(len(df_index))
        t_df = df_index
        window_df = df_index.iloc[0:-1]
        window_df['dis_weight'] = window_df['ii'].map(lambda x: math.log(x + 2) / math.log(max(df_index['ii']) + 1))
        window_df['pri_weight'] = window_df['last'].map(
            lambda x: math.log(t_df.iloc[-1]['last'] / abs(x - t_df.iloc[-1]['last'])))
        numerator = window_df[window_df['last'] > t_df.iloc[-1]['last']][
            ['dis_weight', 'pri_weight', 'volume']].prod(axis=1).sum()
        denominator = window_df[['dis_weight', 'pri_weight', 'volume']].prod(axis=1).sum()
        resistance = numerator / denominator
        return resistance

    def resistance_INSERT(R_Value, profit,R_Value_ETH,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间

        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO resistance (R_Value,profit,R_Value_ETH) VALUES (%s,%s,%s)'
                cursor.execute(sql, (R_Value, profit,R_Value_ETH));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

    def resistance_DATA_Analyse(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                sql = 'SELECT * from Python.resistance where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`)'
                cursor.execute(sql);
                result = cursor.fetchall()
                result = np.array(result)
                columns = [desc[0] for desc in cursor.description]
                data = pd.DataFrame(list(result), columns=columns)
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return data
        finally:
            connection.close();


    def get_data(i,db):
        data = PRE.resistance_DATA_Analyse(1,db)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip(data['ID'], data['R_Value']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter

    def ask_vl_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(bid_ask_vl) from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  bid_ask_vl<0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(bid_ask_vl)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def ask_vl_SELECT_eth(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(bid_ask_vl) from Python.kpi_eth where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  bid_ask_vl<0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(bid_ask_vl)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def bid_vl_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(bid_ask_vl) from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  bid_ask_vl>0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(bid_ask_vl)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def bid_vl_SELECT_eth(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(bid_ask_vl) from Python.kpi_eth where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  bid_ask_vl>0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(bid_ask_vl)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def ask_bar_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(BAR) from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  BAR<0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(BAR)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();
    def ask_bar_SELECT_eth(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(BAR) from Python.kpi_eth where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  BAR<0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(BAR)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def bid_bar_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(BAR) from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  BAR>0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(BAR)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def bid_bar_SELECT_eth(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(BAR) from Python.kpi_eth where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) AND  BAR>0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(BAR)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def volume_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(volume) from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`)'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(volume)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def KPI(d,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间

        d=d+1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT BAR,bid_ask_vl,volume FROM kpi WHERE ID=(SELECT MAX(ID) FROM kpi )'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                AVG_COST=result["BAR"]
                Amount=result["bid_ask_vl"]
                usdt=result["volume"]
                # print(result,AVG_COST,Amount)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return AVG_COST, Amount, usdt
        finally:
            connection.close();
    def KPI_ETH(d,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间

        d=d+1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT BAR,bid_ask_vl,volume FROM kpi_eth WHERE ID=(SELECT MAX(ID) FROM kpi_eth )'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                AVG_COST=result["BAR"]
                Amount=result["bid_ask_vl"]
                usdt=result["volume"]
                # print(result,AVG_COST,Amount)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return AVG_COST, Amount, usdt
        finally:
            connection.close();

    def Trade_INSERT(BAR_AVG,Resistance_AVG,Deep_AVG,BAR_AVG_ETH,Resistance_AVG_ETH,Deep_AVG_ETH,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间

        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO trade (BAR_AVG,Resistance_AVG,Deep_AVG,BAR_AVG_ETH,Resistance_AVG_ETH,Deep_AVG_ETH) VALUES (%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (BAR_AVG,Resistance_AVG,Deep_AVG,BAR_AVG_ETH,Resistance_AVG_ETH,Deep_AVG_ETH));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();


    def resistance_SELECT(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(R_Value) from Python.resistance where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) and R_Value>0'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(R_Value)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def get_data_min(i):
        def resistance_min(i):
            config = {
                'host': 'mysql-roommates.v2.tenxapp.com',
                'port': 54529,
                'user': 'root',
                'password': '1234',
                'db': 'Python',
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
            }
            connection = pymysql.connect(**config)
            # 获取插入的时间
            i = 2
            # 执行sql语句
            try:
                with connection.cursor() as cursor:
                    # 执行sql语句，插入记录
                    # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                    sql = 'SELECT R_Value,R_Value_ETH from Python.resistance where date_sub(now(), INTERVAL 11 MINUTE ) <= updatetime'
                    cursor.execute(sql);
                    result = cursor.fetchall()
                    result = np.array(result)
                    columns = [desc[0] for desc in cursor.description]
                    data = pd.DataFrame(list(result), columns=columns)

                    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                return data
            finally:
                connection.close();
        data = resistance_min(1)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip( data['R_Value'],data['R_Value_ETH']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter

    def get_data_multiply_min(i):
        def resistance_min(i):
            config = {
                'host': 'mysql-roommates.v2.tenxapp.com',
                'port': 54529,
                'user': 'root',
                'password': '1234',
                'db': 'Python',
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
            }
            connection = pymysql.connect(**config)
            # 获取插入的时间
            i = 2
            # 执行sql语句
            try:
                with connection.cursor() as cursor:

                    # 执行sql语句，插入记录
                    # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                    sql = 'SELECT volume,BAR from Python.kpi where date_sub(now(), INTERVAL 21 MINUTE ) <= updatetime '
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    result = np.array(result)
                    columns = [desc[0] for desc in cursor.description]
                    data = pd.DataFrame(list(result), columns=columns)

                    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                return data
            finally:
                connection.close();
        data = resistance_min(1)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip( data['volume'],data['BAR']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter

    def get_data_multiply_hour(i):
        def resistance_min(i):
            config = {
                'host': 'mysql-roommates.v2.tenxapp.com',
                'port': 54529,
                'user': 'root',
                'password': '1234',
                'db': 'Python',
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
            }
            connection = pymysql.connect(**config)
            # 获取插入的时间
            i = 2
            # 执行sql语句
            try:
                with connection.cursor() as cursor:

                    # 执行sql语句，插入记录
                    # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                    sql = 'SELECT volume,BAR from Python.kpi where date_sub(now(), INTERVAL 61 MINUTE ) <= updatetime '
                    # sql = 'SELECT volume,BAR from Python.kpi where updatetime >"2017-07-03 10:23:47" and updatetime <"2017-07-03 12:01:33"'

                    cursor.execute(sql)
                    result = cursor.fetchall()
                    result = np.array(result)
                    columns = [desc[0] for desc in cursor.description]
                    data = pd.DataFrame(list(result), columns=columns)

                    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                return data
            finally:
                connection.close();

        data = resistance_min(1)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip(data['volume'], data['BAR']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter

    def get_data_multiply_min_eth(i):
        def resistance_min(i):
            config = {
                'host': 'mysql-roommates.v2.tenxapp.com',
                'port': 54529,
                'user': 'root',
                'password': '1234',
                'db': 'Python',
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
            }
            connection = pymysql.connect(**config)
            # 获取插入的时间
            i = 2
            # 执行sql语句
            try:
                with connection.cursor() as cursor:
                    # 执行sql语句，插入记录
                    # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                    sql = 'SELECT volume,BAR from Python.kpi_eth where date_sub(now(), INTERVAL 21 MINUTE ) <= updatetime'
                    # sql = 'SELECT volume,BAR from Python.kpi_eth where updatetime >"2017-07-03 22:28:27"'
                    cursor.execute(sql);
                    result = cursor.fetchall()
                    result = np.array(result)
                    columns = [desc[0] for desc in cursor.description]
                    data = pd.DataFrame(list(result), columns=columns)

                    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                return data
            finally:
                connection.close();
        data = resistance_min(1)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip( data['volume'],data['BAR']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter


    def get_data_multiply_hour_eth(i):
        def resistance_min(i):
            config = {
                'host': 'mysql-roommates.v2.tenxapp.com',
                'port': 54529,
                'user': 'root',
                'password': '1234',
                'db': 'Python',
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
            }
            connection = pymysql.connect(**config)
            # 获取插入的时间
            i = 2
            # 执行sql语句
            try:
                with connection.cursor() as cursor:
                    # 执行sql语句，插入记录
                    # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
                    sql = 'SELECT volume,BAR from Python.kpi_eth where date_sub(now(), INTERVAL 61 MINUTE ) <= updatetime'
                    # sql = 'SELECT volume,BAR from Python.kpi_eth where updatetime >"2017-07-03 12:01:33"'
                    cursor.execute(sql);
                    result = cursor.fetchall()
                    result = np.array(result)
                    columns = [desc[0] for desc in cursor.description]
                    data = pd.DataFrame(list(result), columns=columns)

                    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                return data
            finally:
                connection.close();
        data = resistance_min(1)
        X_parameter = []
        Y_parameter = []
        for bar, last in zip( data['volume'],data['BAR']):
            X_parameter.append(float(bar))
            Y_parameter.append(float(last))
        return X_parameter, Y_parameter


    def resistance_SELECT_eth(i,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间
        i = 2

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(R_Value_ETH) from Python.resistance where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`)'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(R_Value_ETH)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

    def KPI_R(d,db):
        config = {
            'host': 'mysql-roommates.v2.tenxapp.com',
            'port': 54529,
            'user': 'root',
            'password': '1234',
            'db': db,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
        }
        connection = pymysql.connect(**config)
        # 获取插入的时间

        d=d+1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT R_Value,R_Value_ETH FROM resistance WHERE ID=(SELECT MAX(ID) FROM resistance )'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                AVG_COST=result["R_Value"]
                AVG_COST_ETH=result["R_Value_ETH"]

                # print(result,AVG_COST,Amount)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return AVG_COST,AVG_COST_ETH
        finally:
            connection.close();

    def Change_Ltc(amount,amount_A,info_usdt,price_buy,info_ltc,R_Value_ETH,info_eth,price_sell_eth,AVG_COST_ETH,charge_usdt_eth,InI_COST,NOW_COST,deep_eth, deep,bar_eth,Annotation):
        if info_usdt < amount * price_buy * info_ltc and R_Value_ETH > 0.1 and bar_eth<3.125:
            p_ltc = info_ltc * amount
            if p_ltc < 0.1:
                p_ltc = 0.1
            print('调整配仓+', p_ltc, 'ltc')

            p_pcgt = (amount * price_buy * info_ltc-info_usdt) / (info_eth * price_sell_eth)
            p_pcgt = math.floor(p_pcgt * 100) / 100
            p_pcg = float('%.2f' % p_pcgt)
            strategy.amount_sell_HV_ETH(info_eth, price_sell_eth, AVG_COST_ETH, charge_usdt_eth, InI_COST, NOW_COST,
                                        deep_eth,
                                        'eth_usdt', 'Python', p_pcg,'调整配仓')
            price_buy = price_buy + 2
            strategy.amount_buy_HV(info_usdt, price_buy, InI_COST, NOW_COST, deep, 'ltc_btc', 'Python', 1,Annotation)
        else:
            strategy.amount_buy_HV(info_usdt, price_buy, InI_COST, NOW_COST, deep, 'ltc_btc', 'Python', amount_A,Annotation)

    def Change_Eth(amount,amount_A,info_usdt,price_buy_eth,info_eth,R_Value,info_ltc,price_sell,AVG_COST,charge_usdt,InI_COST,NOW_COST,deep,deep_eth,bar,Annotation):
        if info_usdt < amount * price_buy_eth * info_eth and R_Value > 0.1 and bar<1.25:
            p_eth = info_eth * amount
            if p_eth < 0.1:
                p_eth = 0.1
            print('调整配仓+', p_eth, 'eth')
            Annotation_Sell = '调整配仓'
            p_pcgt = (amount * price_buy_eth * info_eth-info_usdt) / (info_ltc * price_sell)
            p_pcgt = math.floor(p_pcgt * 100) / 100
            p_pcg = float('%.2f' % p_pcgt)
            strategy.amount_sell_HV(info_ltc, price_sell, AVG_COST, charge_usdt, InI_COST, NOW_COST, deep,
                                    'ltc_btc', 'Python', p_pcg,'调整配仓')
            price_buy_eth = price_buy_eth + 2
            strategy.amount_buy_HV_ETH(info_usdt, price_buy_eth, InI_COST, NOW_COST, deep_eth, 'eth_usdt', 'Python', 1,Annotation)
        else:
            strategy.amount_buy_HV_ETH(info_usdt, price_buy_eth, InI_COST, NOW_COST, deep_eth, 'eth_usdt', 'Python', amount_A,Annotation)
if __name__=='__main__':



    #     print(der_eth)



    #########eth#########




    ######################################################
    InI_COST = mysql.INI_COST(1, 'Python')[0]
    NOW_COST = mysql.mysql_SELECT_BUY_SELL(1, 'Python')[5]