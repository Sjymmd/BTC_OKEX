# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:36:19 2017

"""
# 用于访问OKCOIN 现货REST API
##############################################################################
import http.client
import urllib
import json
import hashlib
import time
import sys

def buildMySign(params, secretKey):
    sign = ''
    for key in sorted(params.keys()):
        sign += key + '=' + str(params[key]) + '&'
    data = sign + 'secret_key=' + secretKey
    return hashlib.md5(data.encode("utf8")).hexdigest().upper()


def httpGet(url, resource, params=''):

    try:

        conn = http.client.HTTPSConnection(url, timeout=10)
        conn.request("GET", resource + '?' + params)
        # print(resource + '?' + params)
        response = conn.getresponse()
        data = response.read().decode('utf8')

    except:

        import requests
        proxies = {'https': 'socks5://127.0.0.1:1080'}
        url_in = 'https://'+url + resource + '?' + params
        while True:
            try:
                response = requests.get(url = url_in,timeout =10, proxies=proxies,verify=False)
                break
            except:
                print('SSL_REQUEST_ERROR')
                time.sleep(5)
                continue
        data = response.content.decode('utf8')

    return json.loads(data)


def httpPost(url, resource, params):
    headers = {
        "Content-type": "application/x-www-form-urlencoded"

    }
    temp_params = urllib.parse.urlencode(params)
    try:

        conn = http.client.HTTPSConnection(url, timeout=10)
        # print("https://"+url+resource+"?"+str(temp_params))
        conn.request("POST", resource, temp_params, headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        params.clear()
        conn.close()
    except:

        import requests
        proxies = {'https': 'socks5://127.0.0.1:1080'}
        while True:
            try:
                response = requests.post('https://' + url + resource, data=temp_params, headers=headers,timeout = 10,proxies=proxies,
                            verify=False)
                break
            except:
                print('SSL_POST_ERROR')
                time.sleep(5)
                continue

        data = response.content.decode('utf8')
        params.clear()
    return data


#############################################################################
import urllib


class OKCoinSpot:
    def __init__(self, url, apikey, secretkey):
        self.__url = url
        self.__apikey = apikey
        self.__secretkey = secretkey
        #print(self.__secretkey)

        # 获取OKCOIN现货行情信息

    def ticker(self, symbol=''):
        TICKER_RESOURCE = "/api/v1/ticker.do"
        params = ''
        if symbol:
            params = 'symbol=%(symbol)s' % {'symbol': symbol}
        return httpGet(self.__url, TICKER_RESOURCE, params)

        # 获取OKCOIN现货市场深度信息


    def depth(self, symbol=''):
        DEPTH_RESOURCE = "/api/v1/depth.do"
        params = ''
        if symbol:
            params = 'symbol=%(symbol)s' % {'symbol': symbol}
        return httpGet(self.__url, DEPTH_RESOURCE, params)

        # 获取OKCOIN现货历史交易信息

    def trades(self, symbol=''):
        TRADES_RESOURCE = "/api/v1/trades.do"
        params = ''
        if symbol:
            params = 'symbol=%(symbol)s' % {'symbol': symbol}
        return httpGet(self.__url, TRADES_RESOURCE, params)

        # 获取用户现货账户信息

    def userinfo(self):
        USERINFO_RESOURCE = "/api/v1/userinfo.do"
        params = {}
        params['api_key'] = self.__apikey
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, USERINFO_RESOURCE, params)

        # 现货交易

    def trade(self, symbol, tradeType, price='', amount=''):
        TRADE_RESOURCE = "/api/v1/trade.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'type': tradeType
        }
        if price:
            params['price'] = price
        if amount:
            params['amount'] = amount
        # params['amount'] = amount
        # params['price'] = price
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, TRADE_RESOURCE, params)

        # 现货批量下单

    def batchTrade(self, symbol, tradeType, orders_data):
        BATCH_TRADE_RESOURCE = "/api/v1/batch_trade.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'type': tradeType,
            'orders_data': orders_data
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, BATCH_TRADE_RESOURCE, params)

        # 现货取消订单

    def cancelOrder(self, symbol, orderId):
        CANCEL_ORDER_RESOURCE = "/api/v1/cancel_order.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'order_id': orderId
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, CANCEL_ORDER_RESOURCE, params)

        # 现货订单信息查询

    def orderinfo(self, symbol, orderId):
        ORDER_INFO_RESOURCE = "/api/v1/order_info.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'order_id': orderId
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, ORDER_INFO_RESOURCE, params)

        # 现货批量订单信息查询

    def ordersinfo(self, symbol, orderId, tradeType):
        ORDERS_INFO_RESOURCE = "/api/v1/orders_info.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'order_id': orderId,
            'type': tradeType
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, ORDERS_INFO_RESOURCE, params)

        # 现货获得历史订单信息

    def orderHistory(self, symbol, status, currentPage, pageLength):
        ORDER_HISTORY_RESOURCE = "/api/v1/order_history.do"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'status': status,
            'current_page': currentPage,
            'page_length': pageLength
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, ORDER_HISTORY_RESOURCE, params)

    def getKline(self, duration, size, since,ltc_cny):
        kline_resourse = "https://www.okex.com/api/v1/kline.do"
        params = {
            # 'api_key': self.__apikey,
            'symbol': ltc_cny,
            'type': duration,
            'size': size,
            'since': since
        }
        temp_params = urllib.parse.urlencode(params)
        # print(temp_params)
        return httpGet(self.__url, kline_resourse, temp_params)

    # def progresive(count):
    #     for i in range(1, count):
    #         sys.stdout.write('>' + '->' + "\b\b")
    #         sys.stdout.flush()
    #         time.sleep(1)
    #     print
    def progresive(count):
        a = 0
        for i in range(1, count):
            a = a + 1
            a = str(a)
            sys.stdout.write(a + ' ' + '->' + "\b\b")
            sys.stdout.flush()
            a = int(a)
            time.sleep(1)
        print
        #############################################################################


# !/usr/bin/python
# -*- coding: utf-8 -*-
# 用于访问OKCOIN 期货REST API
import urllib


class OKCoinFuture:
    def __init__(self, url, apikey, secretkey):
        self.__url = url
        self.__apikey = apikey
        self.__secretkey = secretkey

        # OKCOIN期货行情信息

    def future_ticker(self, symbol, contractType):
        FUTURE_TICKER_RESOURCE = "/api/v1/future_ticker.do"
        params = ''
        if symbol:
            params += '&symbol=' + symbol if params else 'symbol=' + symbol
        if contractType:
            params += '&contract_type=' + contractType if params else 'contract_type=' + symbol
        return httpGet(self.__url, FUTURE_TICKER_RESOURCE, params)

        # OKCoin期货市场深度信息

    def future_depth(self, symbol, contractType, size):
        FUTURE_DEPTH_RESOURCE = "/api/v1/future_depth.do"
        params = ''
        if symbol:
            params += '&symbol=' + symbol if params else 'symbol=' + symbol
        if contractType:
            params += '&contract_type=' + contractType if params else 'contract_type=' + symbol
        if size:
            params += '&size=' + size if params else 'size=' + size
        return httpGet(self.__url, FUTURE_DEPTH_RESOURCE, params)

        # OKCoin期货交易记录信息

    def future_trades(self, symbol, contractType):
        FUTURE_TRADES_RESOURCE = "/api/v1/future_trades.do"
        params = ''
        if symbol:
            params += '&symbol=' + symbol if params else 'symbol=' + symbol
        if contractType:
            params += '&contract_type=' + contractType if params else 'contract_type=' + symbol
        return httpGet(self.__url, FUTURE_TRADES_RESOURCE, params)

        # OKCoin期货指数

    def future_index(self, symbol):
        FUTURE_INDEX = "/api/v1/future_index.do"
        params = ''
        if symbol:
            params = 'symbol=' + symbol
        return httpGet(self.__url, FUTURE_INDEX, params)

        # 获取美元人民币汇率

    def exchange_rate(self):
        EXCHANGE_RATE = "/api/v1/exchange_rate.do"
        return httpGet(self.__url, EXCHANGE_RATE, '')

        # 获取预估交割价

    def future_estimated_price(self, symbol):
        FUTURE_ESTIMATED_PRICE = "/api/v1/future_estimated_price.do"
        params = ''
        if symbol:
            params = 'symbol=' + symbol
        return httpGet(self.__url, FUTURE_ESTIMATED_PRICE, params)

        # 期货全仓账户信息

    def future_userinfo(self):
        FUTURE_USERINFO = "/api/v1/future_userinfo.do?"
        params = {}
        params['api_key'] = self.__apikey
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_USERINFO, params)

        # 期货全仓持仓信息

    def future_position(self, symbol, contractType):
        FUTURE_POSITION = "/api/v1/future_position.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_POSITION, params)

        # 期货下单

    def future_trade(self, symbol, contractType, price='', amount='', tradeType='', matchPrice='', leverRate=''):
        FUTURE_TRADE = "/api/v1/future_trade.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType,
            'amount': amount,
            'type': tradeType,
            'match_price': matchPrice,
            'lever_rate': leverRate
        }
        if price:
            params['price'] = price
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_TRADE, params)

        # 期货批量下单

    def future_batchTrade(self, symbol, contractType, orders_data, leverRate):
        FUTURE_BATCH_TRADE = "/api/v1/future_batch_trade.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType,
            'orders_data': orders_data,
            'lever_rate': leverRate
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_BATCH_TRADE, params)

        # 期货取消订单

    def future_cancel(self, symbol, contractType, orderId):
        FUTURE_CANCEL = "/api/v1/future_cancel.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType,
            'order_id': orderId
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_CANCEL, params)

        # 期货获取订单信息

    def future_orderinfo(self, symbol, contractType, orderId, status, currentPage, pageLength):
        FUTURE_ORDERINFO = "/api/v1/future_order_info.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType,
            'order_id': orderId,
            'status': status,
            'current_page': currentPage,
            'page_length': pageLength
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_ORDERINFO, params)

        # 期货逐仓账户信息

    def future_userinfo_4fix(self):
        FUTURE_INFO_4FIX = "/api/v1/future_userinfo_4fix.do?"
        params = {'api_key': self.__apikey}
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_INFO_4FIX, params)

        # 期货逐仓持仓信息

    def future_position_4fix(self, symbol, contractType, type1):
        FUTURE_POSITION_4FIX = "/api/v1/future_position_4fix.do?"
        params = {
            'api_key': self.__apikey,
            'symbol': symbol,
            'contract_type': contractType,
            'type': type1
        }
        params['sign'] = buildMySign(params, self.__secretkey)
        return httpPost(self.__url, FUTURE_POSITION_4FIX, params)

    def getKline(self, duration, size, since):
        kline_resourse = "https://www.okcoin.com/api/v1/future_kline.do?"
        params = {
            # 'api_key': self.__apikey,
            'symbol': "ltc_usd",
            'type': duration,
            'contract_type': "quarter",
            'size': size,
            'since': since
        }
        temp_params = urllib.parse.urlencode(params)
        return httpGet(self.__url, kline_resourse, temp_params)
        # return httpPost(self.__url,kline_resourse,params)
        # temp_params = urllib.parse.urlencode(params)
        # print(temp_params)
        # return httpGet(self.__url, kline_resourse, temp_params)

########################
