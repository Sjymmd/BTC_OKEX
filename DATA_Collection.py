# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:36:19 2017

@author: mjc

E-mail:1005965744@qq.com

Differentiate yourself in the world from anyone else.
"""
#####################################################
from OkcoinSpotAPI import *
from Mysql import *
from strategy import *
import pandas as pd
from Analyse import *
# from PRE import *
import numpy as np
import datetime
import time
import math

######################################################
okcoinRESTURL = 'www.okex.com'
apikey='1f74f0f9-54c2-4e1c-b653-3b3d2b2995d8'
secretkey='A04BBDEDC2B0B4436D853AA90BD4DD2B'
okcoinSpot = OKCoinSpot(okcoinRESTURL, apikey, secretkey)
######################################################
try:
    # print(okcoinSpot.userinfo())
    info = initial.info_get(1)
    (info_ltc, info_total, info_cny, info_eth) = (info[0], info[1], info[2], info[3])
    (info_freeze_cny, info_freeze_ltc) = (info[4], info[5])
    global true
    true = ''
    global false
    false = ''
except:
    print('info Error')
try:
    ref_upper = initial.boll_get(1, 'ltc_btc')[0]
    ref_lower = initial.boll_get(1, 'ltc_btc')[1]
    ref_close = initial.boll_get(1, 'ltc_btc')[2]
    ref_ma34 = initial.boll_get(1, 'ltc_btc')[3]
    last = okcoinSpot.ticker('ltc_btc')
    last = float(last["ticker"]["last"])

    ref_upper_eth = initial.boll_get(1, 'eth_usdt')[0]
    ref_lower_eth = initial.boll_get(1, 'eth_usdt')[1]
    ref_close_eth = initial.boll_get(1, 'eth_usdt')[2]
    ref_ma34_eth = initial.boll_get(1, 'eth_usdt')[3]

    last_eth = okcoinSpot.ticker('eth_usdt')
    last_eth = float(last_eth["ticker"]["last"])
except:
    print('boll Error')
try:
    # last = okcoinSpot.ticker('ltc_btc')
    # last = float(last["ticker"]["last"])
    print(datetime.datetime.now())
    print('Info_Total', info_total, 'Info_Ltc', info_ltc, 'Info_eth', info_eth, 'Info_cny', info_cny)
    print('Info_Freeze_Ltc', info_freeze_ltc, 'Info_Freeze_Cny', info_freeze_cny)
    print('ltc')
    print('ref_close', ref_close, 'ref_upper', ref_upper, 'ref_ma34', ref_ma34)
    print('ref_lower', ref_lower, 'last', last)
    print('eth')
    print('ref_close', ref_close_eth, 'ref_upper', ref_upper_eth, 'ref_ma34', ref_ma34_eth)
    print('ref_lower', ref_lower_eth, 'last', last_eth)
except:
    print('First_Step Error')
# time.sleep(58)

OKCoinSpot.progresive(6)
print('\n')
i = 0
while True:

    #######INFO_GET######
    #########ltc#########
    try:

        R_AVG = PRE.resistance_SELECT(1, 'Python')
        # print(R_AVG)
        # except:
        #     print('API ERRPR')
        #     continue
        x, y, z = PRE.KPI(1, 'Python')
        bar = x
        x = float(x)
        y = float(y)
        if x < 0:
            BAR_AVG = float(PRE.ask_bar_SELECT(1, 'Python'))
        else:
            BAR_AVG = float(PRE.bid_bar_SELECT(1, 'Python'))
        if y < 0:
            bid_ask_vl = float(PRE.ask_vl_SELECT(1, 'Python'))
        else:
            bid_ask_vl = float(PRE.bid_vl_SELECT(1, 'Python'))
    except:
        print('R_AVG ERROR_ltc')
        continue
    #########eth#########
    try:

        R_AVG_ETH = PRE.resistance_SELECT_eth(1, 'Python')
        # print(R_AVG)
        # except:
        #     print('API ERRPR')
        #     continue
        x, y, z = PRE.KPI_ETH(1, 'Python')
        bar_eth = x
        x = float(x)
        y = float(y)
        if x < 0:
            BAR_AVG_ETH = float(PRE.ask_bar_SELECT_eth(1, 'Python'))
        else:
            BAR_AVG_ETH = float(PRE.bid_bar_SELECT_eth(1, 'Python'))
        if y < 0:
            bid_ask_vl_eth = float(PRE.ask_vl_SELECT_eth(1, 'Python'))
        else:
            bid_ask_vl_eth = float(PRE.bid_vl_SELECT_eth(1, 'Python'))
    except:
        print('R_AVG ERROR_eth')
        continue

    try:

        try:
            info = initial.info_get(1)
            (info_ltc, info_total, info_cny, info_eth) = (info[0], info[1], info[2], info[3])
            (info_freeze_cny, info_freeze_ltc) = (info[4], info[5])
        except:
            continue
        #########ltc#########
        try:
            deep = pd.DataFrame(okcoinSpot.depth('ltc_btc'))
            deep = analyse.cut(deep)
            price_buy = float(str(deep['bid_price'].max()))
            price_sell = float(str(deep['bid_price'][1]))
            # BID_AVG =strategy.bid_ask_vol_diff(deep)[0]
            # INSERT_TMP=strategy.bid_ask_vol_diff(deep)[1]
            X, Y = strategy.bid_ask_vol_diff(deep, 'Python')
            BID_AVG = X
            INSERT_TMP = Y

        except:
            print('info_error')
            continue

        #########eth#########
        try:
            deep_eth = pd.DataFrame(okcoinSpot.depth('eth_usdt'))
            deep_eth = analyse.cut(deep_eth)
            price_buy_eth = float(str(deep_eth['bid_price'].max()))
            price_sell_eth = float(str(deep_eth['bid_price'][1]))
            # BID_AVG =strategy.bid_ask_vol_diff(deep)[0]
            # INSERT_TMP=strategy.bid_ask_vol_diff(deep)[1]
            XX, YY = strategy.bid_ask_vol_diff_eth(deep_eth, 'Python')
            BID_AVG_ETH = XX
            INSERT_TMP_ETH = YY

        except:
            print('info_error_2')
            continue
        ######################################################
        try:
            #########ltc#########
            last = okcoinSpot.ticker('ltc_btc')
            last = float(last["ticker"]["last"])
            boll = strategy.bbands_MA(94, 2, 'ltc_btc')
            upper = boll[0]
            lower = boll[2]
            close = okcoinSpot.getKline('1min', '1', '0', 'ltc_btc')[0][4]
            ma34 = pd.DataFrame(okcoinSpot.getKline('1min', '34', '0', 'ltc_btc')).iloc[::, 4].mean()

            #########eth#########
            last_eth = okcoinSpot.ticker('eth_usdt')
            last_eth = float(last_eth["ticker"]["last"])
            boll_eth = strategy.bbands_MA(94, 2, 'eth_usdt')
            upper_eth = boll_eth[0]
            lower_eth = boll_eth[2]
            close_eth = okcoinSpot.getKline('1min', '1', '0', 'eth_usdt')[0][4]
            ma34_eth = pd.DataFrame(okcoinSpot.getKline('1min', '34', '0', 'eth_usdt')).iloc[::, 4].mean()
            ######################################################
            InI_COST = mysql.INI_COST(1, 'Python')[0]
            NOW_COST = mysql.mysql_SELECT_BUY_SELL(1, 'Python')[5]
        except:
            print('get_ERROR')
            continue
        #########ltc#########
        AVG_COST = mysql.mysql_SELECT_BUY_SELL(1, 'Python')[0]
        charge_cny = last * 0.1 * 0.002
        # NOW_COST = mysql.mysql_SELECT_BUY_SELL(1,'Python')[0] * mysql.mysql_SELECT_BUY_SELL(1,'Python')[1] + \
        #            mysql.mysql_SELECT_BUY_SELL(1,'Python')[2]
        try:
            R_Value = float(PRE.resistance(1, 'Python'))
        except:
            print('R_ERROR')
            continue
        #########eth#########
        AVG_COST_ETH = mysql.mysql_SELECT_BUY_SELL(1, 'Python')[4]
        charge_cny_eth = last_eth * 0.01 * 0.001
        try:
            R_Value_ETH = float(PRE.resistance_eth(1, 'Python'))
        except:
            print('R_ERROR_ETH')
            continue
        ######################################################

        NOWCOST = last * mysql.mysql_SELECT_BUY_SELL(1, 'Python')[1] + mysql.mysql_SELECT_BUY_SELL(1, 'Python')[
            2] + last_eth * mysql.mysql_SELECT_BUY_SELL(1, 'Python')[3]

        NOWCOST_LTC = last * mysql.mysql_SELECT_BUY_SELL(1, 'Python')[1]

        NOWCOST_ETH = last_eth * mysql.mysql_SELECT_BUY_SELL(1, 'Python')[3]

        # print(NOWCOST)

        try:
            PRE.Trade_INSERT(BAR_AVG, R_AVG, bid_ask_vl, BAR_AVG_ETH, R_AVG_ETH, bid_ask_vl_eth, 'Python')
        except:
            print('INSERT Trade Fail')
            continue

        #########Total#########
        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now,i)
        print('Info_Total', info_total, 'Info_Ltc', info_ltc, 'Info_eth', info_eth, 'Info_cny', info_cny)
        print('Info_Freeze_Ltc', info_freeze_ltc, 'Info_Freeze_Cny', info_freeze_cny)

        #########ltc#########

        if R_Value == 0.0:
            R_Value = PRE.KPI_R(1, 'Python')[0]
        print('LTC')
        print(R_Value, NOWCOST_LTC)

        print(PRE.DATA(1, 'Python', 'ltc_btc', INSERT_TMP))
        print('R_AVG', R_AVG, 'BAR_AVG', BAR_AVG, 'bid_ask_vl', bid_ask_vl)

        print('close', close, 'upper', upper, 'ma34', ma34, 'reclose', ref_close, 'ref_ma34', ref_ma34)
        # print('close',close,'upper',upper,'ma34',ma34)
        print('lower', lower, 'last', last, 'SELL_FEILD', last - AVG_COST - charge_cny / 0.1 - 7, 'charge_cny',
              charge_cny)
        print('bid_ask_vol', INSERT_TMP, 'bid_ask_avg', BID_AVG, 'bid_ask_price',
              analyse.bid_ask_price_diff(deep))
        # time.sleep(59)
        #########eth#########
        if R_Value_ETH == 0.0:
            R_Value_ETH = PRE.KPI_R(1, 'Python')[1]

        print('ETH')
        print(R_Value_ETH, NOWCOST_ETH)
        print(PRE.DATA_eth(1, 'Python', 'eth_usdt', INSERT_TMP_ETH))
        print(R_AVG_ETH, BAR_AVG_ETH, bid_ask_vl_eth)

        print('close', close_eth, 'upper', upper_eth, 'ma34', ma34_eth, 'reclose', ref_close_eth, 'ref_ma34',
              ref_ma34_eth)
        # print('close',close,'upper',upper,'ma34',ma34)
        print('lower', lower_eth, 'last', last_eth, 'SELL_FEILD', last_eth - AVG_COST_ETH - charge_cny_eth / 0.1 - 70,
              'charge_cny',
              charge_cny_eth)
        print('bid_ask_vol', INSERT_TMP_ETH, 'bid_ask_avg', BID_AVG_ETH, 'bid_ask_price',
              analyse.bid_ask_price_diff(deep_eth))
        # OKCoinSpot.progresive(61)
        ######################################################

    except:
        print('resistance error')
        continue
    PRE.resistance_INSERT(R_Value, info_total, R_Value_ETH, 'Python')
    print(price_buy, price_sell, price_buy_eth, price_sell_eth)

    ######################################################
    #######resistence sell warning#######
    XXX, YYY = PRE.get_data_min(1)
    # X = float(PRE.get_data_min(1)[0])
    ZZZ = np.arange(1, len(XXX) + 1)
    poly = np.polyfit(ZZZ, XXX, 1)
    der = np.polyder(poly)
    poly_eth = np.polyfit(ZZZ, YYY, 1)
    der_eth = np.polyder(poly_eth)

    #######multiply buy point#######

    #########ltc#########

    volume_min, BAR_min = PRE.get_data_multiply_min(1)
    x_volume = np.arange(1, len(volume_min) + 1)
    poly_point = np.polyfit(x_volume, volume_min, 1)
    der_point = np.polyder(poly_point)
    der_point_abs = abs(der_point)
    # print(der_point)
    volume_min_h, BAR_min_h = PRE.get_data_multiply_hour(1)
    x_volume_h = np.arange(1, len(volume_min_h) + 1)
    poly_point_h = np.polyfit(x_volume_h, volume_min_h, 1)
    der_point_h = np.polyder(poly_point_h)
    der_point_h_abs = abs(der_point_h)

    SellEst_tmp = float('%.2f' % Mysql_KPI.HIGH_Lock(AVG_COST, upper, lower)[0])
    SellEst_tmp_low = float('%.2f' % Mysql_KPI.HIGH_Lock(AVG_COST, upper, lower)[1])
    R_Value_C = float('%.2f' % R_Value)
    SellEst = float(SellEst_tmp * (1 + (1 - R_Value_C) * 0.1))
    SellEst_L = float(SellEst_tmp_low * (1 - (1 - R_Value_C) * 0.1))
    #########eth#########

    volume_min_eth, BAR_min_eth = PRE.get_data_multiply_min_eth(1)
    x_volume_eth = np.arange(1, len(volume_min_eth) + 1)
    poly_point_eth = np.polyfit(x_volume_eth, volume_min_eth, 1)
    der_point_eth = np.polyder(poly_point_eth)
    der_point_eth_abs = abs(der_point_eth)

    volume_hour_eth, BAR_min_eth_h = PRE.get_data_multiply_hour_eth(1)
    x_volume_eth_h = np.arange(1, len(volume_hour_eth) + 1)
    poly_point_eth_h = np.polyfit(x_volume_eth_h, volume_hour_eth, 1)
    der_point_eth_h = np.polyder(poly_point_eth_h)
    der_point_eth_h_abs = abs(der_point_eth_h)

    SellEst_ETH_tmp = float('%.2f' % Mysql_KPI.HIGH_Lock_eth(AVG_COST_ETH, upper_eth, lower_eth)[0])
    SellEst_ETH_tmp_low = float('%.2f' % Mysql_KPI.HIGH_Lock_eth(AVG_COST_ETH, upper_eth, lower_eth)[1])
    R_Value_ETH_C = float('%.2f' % R_Value_ETH)
    SellEst_ETH = float(SellEst_ETH_tmp * (1 + (1 - R_Value_ETH_C) * 0.1))
    SellEst_ETH_L = float(SellEst_ETH_tmp_low * (1 - (1 - R_Value_ETH_C) * 0.1))
    print('SellEst', SellEst, 'SellEst_ETH', SellEst_ETH, 'SellEst_L', SellEst_L, 'SellEst_ETH_L', SellEst_ETH_L)
    print('der_ltc_R', der, 'der_eth_R', der_eth)
    print('der_ltc', der_point, 'der_eth', der_point_eth, 'der_ltc_h', der_point_h, 'der_eth_h', der_point_eth_h)
    # print(der_point_abs)
    #########ltc#########
    try:
        volume = PRE.KPI(1, 'Python')[2]
        DataIn = [bar, volume, INSERT_TMP, R_Value, bar / BAR_AVG, INSERT_TMP / bid_ask_vl, R_Value / R_AVG]
        #        sell,buy=Sklearn.Sell_Clf(DataIn)
        sell = 0
        buy = 0
        print('sell', sell, 'buy', buy)

        #########eth#########

        volume_eth = PRE.KPI_ETH(1, 'Python')[2]
        DataIn_Eth = [bar_eth, volume_eth, INSERT_TMP_ETH, R_Value_ETH, bar_eth / BAR_AVG_ETH,
                      INSERT_TMP_ETH / bid_ask_vl_eth, R_Value_ETH / R_AVG_ETH]
        #        sell_eth,buy_eth =Sklearn.Sell_Clf_Eth(DataIn_Eth)
        sell_eth = 0
        buy_eth = 0
        print('sell_eth', sell_eth, 'buy_eth', buy_eth)
    except:
        print('LTF Error')
        continue

    #########ltc#########
    Feature.Feature_Insert(float(ref_upper), float(ref_lower), float(ref_ma34), float(upper), float(lower), float(ma34),
                           float(last), float(R_AVG), float(BAR_AVG), float(INSERT_TMP),
                           float(bar), float(analyse.bid_ask_price_diff(deep)), float(BID_AVG), float(R_Value),
                           float(der), float(der_point_abs), float(der_point_h_abs), float(volume), 'Python')

    #########eth#########
    Feature.Feature_Insert_eth(float(ref_upper_eth), float(ref_lower_eth), float(ref_ma34_eth), float(upper_eth),
                               float(lower_eth), float(ma34_eth), float(last_eth), float(R_AVG_ETH), float(BAR_AVG_ETH),
                               float(INSERT_TMP_ETH),
                               float(bar_eth), float(analyse.bid_ask_price_diff(deep_eth)), float(BID_AVG_ETH),
                               float(R_Value_ETH), float(der_eth), float(der_point_eth_abs), float(der_point_eth_h_abs),
                               float(volume_eth), 'Python')


    OKCoinSpot.progresive(59)
    #     print('\n')
    try:
        ref_boll = boll
        ref_upper = upper
        ref_lower = lower
        ref_close = close
        ref_ma34 = ma34

        ref_boll_eth = boll_eth
        ref_upper_eth = upper_eth
        ref_lower_eth = lower_eth
        ref_ma34_eth = ma34_eth
        ref_close_eth = close_eth
    except:
        print('json错误4')
        continue

    i = i + 1


