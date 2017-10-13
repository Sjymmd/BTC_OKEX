# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:36:19 2017

@author: mjc

E-mail:1005965744@qq.com

Differentiate yourself in the world from anyone else.
"""
##############################################################################
import pymysql.cursors
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
class mysql:
    def mysql_insert_BUY_IN(LTC_FEE,COST,db):
        # Connect to the database
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
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO BUY_IN (LTC_FEE, COST, UPDATETIME) VALUES (%s, %s, %s)'
                cursor.execute(sql, (LTC_FEE, COST, c));
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()

        finally:
            connection.close();

    def mysql_insert_SELL_OUT(CNY_FEE,PROFIT,db,Annotation):
        # Connect to the database
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
        c = datetime.now()
        # print(c)

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO SELL_OUT (CNY_FEE,PROFIT,UPDATETIME,Annotation) VALUES (%s, %s, %s,%s)'
                cursor.execute(sql, (CNY_FEE,PROFIT,c,Annotation));
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()

        finally:
            connection.close();

    def mysql_insert_TRADE(SUM_FEE,SUM_PROFIT,SUM_COST,db):
        # Connect to the database
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
        c = datetime.now()

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO trade (SUM_FEE,SUM_PROFIT,SUM_COST,UPDATETIME) VALUES (%s, %s, %s,%s)'
                cursor.execute(sql, (SUM_FEE,SUM_PROFIT,SUM_COST,c));
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()

        finally:
            connection.close();

    def mysql_INSERT_BUY_SELL(Amount,Price,FEE,LTC,AVG_COST,CNY,ETH,AVG_COST_ETH,db,Annotation):
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
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO buy_sell (Amount,Price,FEE,LTC,AVG_COST,UPDATETIME,CNY,ETH,AVG_COST_ETH,Annotation) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (Amount,Price,FEE,LTC,AVG_COST,c,CNY,ETH,AVG_COST_ETH,Annotation));
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()

        finally:
            connection.close();

    def mysql_INSERT_BUY_SELL_ETH(Amount,Price,FEE,LTC,AVG_COST,CNY,ETH,AVG_COST_ETH,db,Annotation):
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
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO buy_sell (Amount,Price_ETH,FEE,LTC,AVG_COST,UPDATETIME,CNY,ETH,AVG_COST_ETH,Annotation) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (Amount,Price,FEE,LTC,AVG_COST,c,CNY,ETH,AVG_COST_ETH,Annotation));
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()

        finally:
            connection.close();

    def mysql_SELECT_BUY_SELL(d,db):
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
        c = datetime.now()
        d=d+1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT AVG_COST,LTC,CNY,ETH,AVG_COST_ETH FROM buy_sell WHERE ID=(SELECT MAX(ID) FROM buy_sell)'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                AVG_COST=result["AVG_COST"]
                Amount=result["LTC"]
                CNY=result["CNY"]
                ETH=result["ETH"]
                AVG_COST_ETH=result["AVG_COST_ETH"]
                NOW_COST = result["LTC"] * result["AVG_COST"] + result["CNY"] + result["ETH"] * result["AVG_COST_ETH"]
                # print(result,AVG_COST,Amount)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return AVG_COST, Amount, CNY,ETH,AVG_COST_ETH,NOW_COST
        finally:
            connection.close();

    def COST_CALCUATION(last,amount,FEE,LTC_I,ETH,CNY,db,Annotation):
        try:
            AVG_COST=mysql.mysql_SELECT_BUY_SELL(1,db)[0]
            LTC=mysql.mysql_SELECT_BUY_SELL(1,db)[1]
            # Amount取mysql.mysql_SELECT_BUY_SELL_LTC
            Amount_New=amount+LTC
            if AVG_COST > 0:
                AVG_COST_New=(AVG_COST*LTC+FEE+last*amount)/Amount_New
            else:
                AVG_COST_New=(last*LTC+FEE+last*amount)/Amount_New
            # print(AVG_COST,LTC,Amount_New,AVG_COST_New)
            # CNY=mysql.mysql_SELECT_BUY_SELL(1,db)[2]-amount*last
            AVG_COST_ANO=mysql.mysql_SELECT_BUY_SELL(1,db)[4]
            mysql.mysql_INSERT_BUY_SELL(amount,last,FEE,LTC_I,AVG_COST_New,CNY,ETH,AVG_COST_ANO,db,Annotation)

        except:
            print("JSON错误_COST_CALCUATION")

    def COST_CALCUATION_ETH(last,amount,FEE,LTC_I,ETH,CNY,db,Annotation):
        try:
            AVG_COST=mysql.mysql_SELECT_BUY_SELL(1,db)[4]
            LTC=mysql.mysql_SELECT_BUY_SELL(1,db)[3]
            # Amount取mysql.mysql_SELECT_BUY_SELL_LTC
            Amount_New=amount+LTC
            if AVG_COST > 0:
                AVG_COST_New=(AVG_COST*LTC+FEE+last*amount)/Amount_New
            else:
                AVG_COST_New=(last*LTC+FEE+last*amount)/Amount_New
            # print(AVG_COST,LTC,Amount_New,AVG_COST_New)
            # AVG_COST, Amount, CNY, ETH, AVG_COST_ETH, NOW_COST
            AVG_COST_ANO=mysql.mysql_SELECT_BUY_SELL(1,db)[0]
            # LTC_ANO=mysql.mysql_SELECT_BUY_SELL(1,db)[1]
            # CNY=mysql.mysql_SELECT_BUY_SELL(1,db)[2]-amount*last
            mysql.mysql_INSERT_BUY_SELL_ETH(amount,last,FEE,LTC_I,AVG_COST_ANO,CNY,ETH,AVG_COST_New,db,Annotation)

        except:
            print("JSON错误_COST_CALCUATION")

    def COST_CALCUATION_SELL(last,amount,FEE,LTC_I,ETH,CNY,db,Annotation):
        try:
            AVG_COST=mysql.mysql_SELECT_BUY_SELL(1,db)[0]
            LTC=mysql.mysql_SELECT_BUY_SELL(1,db)[1]
            # Amount_New=LTC-amount
            AVG_COST_New=AVG_COST+FEE/LTC
            AVG_COST_ANO = mysql.mysql_SELECT_BUY_SELL(1, db)[4]
            # CNY=mysql.mysql_SELECT_BUY_SELL(1,db)[2]+amount*last-FEE
            mysql.mysql_INSERT_BUY_SELL(-amount,last,FEE,LTC_I,AVG_COST_New,CNY,ETH,AVG_COST_ANO,db,Annotation)
        except:
            print("COST_CALCUATION_SELL错误")

    def COST_CALCUATION_SELL_ETH(last,amount,FEE,LTC_I,ETH,CNY,db,Annotation):
        try:
            AVG_COST=mysql.mysql_SELECT_BUY_SELL(1,db)[4]
            LTC=mysql.mysql_SELECT_BUY_SELL(1,db)[3]
            # Amount_New=LTC-amount
            AVG_COST_New=AVG_COST+FEE/LTC
            # CNY=mysql.mysql_SELECT_BUY_SELL(1,db)[2]+amount*last-FEE
            AVG_COST_ANO = mysql.mysql_SELECT_BUY_SELL(1, db)[0]
            mysql.mysql_INSERT_BUY_SELL_ETH(-amount,last,FEE,LTC_I,AVG_COST_ANO,CNY,ETH,AVG_COST_New,db,Annotation)
        except:
            print("COST_CALCUATION_SELL错误")

    def SELL_OUT(n_days,db):
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
        c = datetime.now()+timedelta(seconds=2)
        d= (datetime.now()-timedelta(days=n_days))
        # print(c,d)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT sum(PROFIT) FROM SELL_OUT WHERE UPDATETIME BETWEEN %s AND %s'
                cursor.execute(sql,(d,c))
                # 获取查询结果
                result = cursor.fetchone()
                Sum_Profit=result["sum(PROFIT)"]
                # print(result)
                # print(result,AVG_COST,Amount)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return Sum_Profit,d,c
        finally:
            connection.close();

    def INI_COST(i,db):
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
        i=2
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT LTC ,CNY,AVG_COST,ETH,AVG_COST_ETH from  buy_sell where date_sub(curdate(), INTERVAL 30 DAY) <= date(`UPDATETIME`) ORDER BY ID  limit 1'
                cursor.execute(sql);
                result = cursor.fetchone()
                INI_COST=result["LTC"]*result["AVG_COST"]+result["CNY"]+result["ETH"]*result["AVG_COST_ETH"]
                INI_AVG=result["AVG_COST"]
                INI_AVG_ETH=result["AVG_COST_ETH"]

                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return INI_COST,INI_AVG,INI_AVG_ETH
        finally:
            connection.close();

    def BID_AVG_INSERT(BID_AVG,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO trade (BID_AVG,UPDATETIME) VALUES (%s,%s)'
                cursor.execute(sql, (BID_AVG, c));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();
    def ASK_AVG_INSERT(ASK_AVG,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO trade (ASK_AVG,UPDATETIME) VALUES (%s,%s)'
                cursor.execute(sql, (ASK_AVG, c));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

    def BID_AVG_SELECT(i,db):
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
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(BID_AVG) from Python.trade where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`UPDATETIME`) '
                cursor.execute(sql);
                result = cursor.fetchone()
                BID_AVG=result["avg(BID_AVG)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return BID_AVG
        finally:
            connection.close();

    def ASK_AVG_SELECT(i,db):
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
        c = datetime.now()
        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'SELECT avg(ASK_AVG) from Python.trade where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`UPDATETIME`)'
                cursor.execute(sql);
                result = cursor.fetchone()
                ASK_AVG=result["avg(ASK_AVG)"]
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return ASK_AVG
        finally:
            connection.close();

class Mysql_KPI:

    def EMA_SELECT(d,db):
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
        d = d + 1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume FROM kpi WHERE ID=(SELECT MAX(ID) FROM kpi)'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                RE_EMA_12 = result["EMA_12"]
                RE_EMA_26 = result["EMA_26"]
                RE_DEA=result["DEA"]
                RE_DIF=result["DIF"]
                RE_BAR=result["BAR"]
                last=result["last"]
                high=result["high"]
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return RE_EMA_12,RE_EMA_26,RE_DEA,RE_DIF,RE_BAR,last,high
        finally:
            connection.close();

    def EMA_SELECT_ETH(d,db):
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
        d = d + 1
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，进行查询
                sql = 'SELECT EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume FROM kpi_eth WHERE ID=(SELECT MAX(ID) FROM kpi_eth)'
                cursor.execute(sql)
                # 获取查询结果
                result = cursor.fetchone()
                RE_EMA_12 = result["EMA_12"]
                RE_EMA_26 = result["EMA_26"]
                RE_DEA=result["DEA"]
                RE_DIF=result["DIF"]
                RE_BAR=result["BAR"]
                last=result["last"]
                high=result["high"]
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return RE_EMA_12,RE_EMA_26,RE_DEA,RE_DIF,RE_BAR,last,high
        finally:
            connection.close();

    def EMA_INSERT(EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO kpi (EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

    def EMA_INSERT_ETH(EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO kpi_eth (EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (EMA_12, EMA_26,last, high, low,DEA,DIF,BAR,bid_ask_vl,volume));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

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
        c = datetime.now()
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
        c = datetime.now()
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
        c = datetime.now()
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
        c = datetime.now()
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

    def DATA_Analyse(i,db):
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
                sql = 'SELECT * from Python.kpi where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) '
                cursor.execute(sql);
                result = cursor.fetchall()
                result=np.array(result)
                columns=[desc[0] for desc in cursor.description]
                data= pd.DataFrame(list(result), columns=columns)
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return data
        finally:
            connection.close();

    def DATA_Analyse_eth(i,db):
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
                sql = 'SELECT * from Python.kpi_eth where date_sub(curdate(), INTERVAL 24 HOUR ) <= date(`updatetime`) '
                cursor.execute(sql);
                result = cursor.fetchall()
                result=np.array(result)
                columns=[desc[0] for desc in cursor.description]
                data= pd.DataFrame(list(result), columns=columns)
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return data
        finally:
            connection.close();

    def HIGH_Lock(last,high,low):
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

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'

                sql = 'SELECT avg(high),avg(low) from Python.kpi where round(last)=round(%s) '

                cursor.execute(sql,last);
                result = cursor.fetchone()

                High=result["avg(high)"]
                if High is None:
                    High=high
                Low=result["avg(low)"]
                if Low is None:
                    Low=low

                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return High,Low
        finally:
            connection.close();

    def HIGH_Lock_eth(last,high,low):
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

        # 执行sql语句
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'

                sql = 'SELECT avg(high),avg(low) from Python.kpi_eth where round(last)=round(%s) '

                cursor.execute(sql,last);
                result = cursor.fetchone()
                Low=result["avg(low)"]
                High=result["avg(high)"]
                if High is None:
                    High=high
                if Low is None:
                    Low = low
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
            return High,low
        finally:
            connection.close();

class Feature:

    def Feature_Insert(ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO feature (ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

    def Feature_Insert_eth(ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume,db):
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
        c = datetime.now()
        # print(c)
        try:
            with connection.cursor() as cursor:
                # 执行sql语句，插入记录
                sql = 'INSERT INTO feature_eth (ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cursor.execute(sql, (ref_upper,ref_lower,ref_ma34,upper,lower,ma34,last,R_AVG,BAR_AVG,bid_ask_vl,bar,deep,bid_ask_avg,R_Value,der,der_point_abs,der_point_h_abs,volume));
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                connection.commit()
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
            connection.commit()
        finally:
            connection.close();

if __name__=='__main__':

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
    # 执行sql语句
    try:
        with connection.cursor() as cursor:
            # 执行sql语句，插入记录
            # sql = 'SELECT * from Python.kpi ORDER BY  ID DESC limit 0,20'
            sql = 'SELECT * from Python.kpi_eth'
            cursor.execute(sql);
            result = cursor.fetchall()
            result = np.array(result)
            columns = [desc[0] for desc in cursor.description]
            data = pd.DataFrame(list(result), columns=columns)
            # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
        connection.commit()
        print(data)

    finally:
        connection.close();
