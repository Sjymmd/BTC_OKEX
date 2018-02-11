from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import matplotlib.pyplot as plt

Okex_Api = Okex_Api()
# Coin = Okex_Api.GetCoin()
Coin = ['btc_usdt','snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChose = '1 day'
Okex_Api._Lenth = 24*100
for x in Coin[:int(Okex_Api._CoinLenth)]:
    DataFrame = pd.DataFrame(columns=("Coin", "Cny", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
    data = pd.DataFrame(okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChose], Okex_Api._Lenth, '0', x)).iloc[:Okex_Api._Lenth-1, ]
    Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
    price = float(data.iloc[0, 4])
    Cny = round(price * Okex_Api._USDT_CNY, 2)
    Volume = float(data.iloc[0, 5])
    Volume_Mean = Volume
    Volume_Pre = round(Volume / 1000, 2)
    Volume_Pre_P = 0
    if Volume_Mean == 0:
        Volume_Inc = 0
    else:
        Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
    Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                           'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
    DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
    for lenth in range(1,Okex_Api._Lenth-1):
        try:
            Increase = (float(data.iloc[lenth, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
            Increase = str('%.2f' % (Increase) + '%')
            price = float(data.iloc[lenth, 4])
            Cny = round(price * Okex_Api._USDT_CNY, 2)
            Volume = data.iloc[:lenth+1, 5].apply(pd.to_numeric)
            Volume_Mean = round(Volume.mean() / 1000, 2)
            Volume_Pre = round(Volume.iloc[lenth] / 1000, 2)
            Volume_Pre_P = round((Volume[lenth] / Volume[lenth - 1])-1, 2)
            Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
            Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                                   'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
            DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        except:
            break
    # print(DataFrame)
    Coin_Snt= plt.figure('%s'%x,figsize=(12,6 ))
    plt.grid()
    ax1 = plt.subplot(311)
    ax1.xaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='major')
    line1=ax1.plot(DataFrame['Cny'], linewidth=1.0, color='blue')
    ax1.legend(line1, ('%s-%d%s'%(x,Okex_Api._Lenth,str(Okex_Api._KlineChose)[1:]),))
    ax2 = plt.subplot(312)
    ax2.xaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='major')
    line2=ax2.bar(np.arange(len(DataFrame['_VolumeS'])),DataFrame['_VolumeS'], color='orange')
    ax2.legend(line2, ('%s'%DataFrame.columns[5],))
    ax3= plt.subplot(313)
    ax3.xaxis.grid(True, which='major')
    ax3.yaxis.grid(True, which='major')
    line3 = ax3.bar(np.arange(len(DataFrame['_VolumeM'])),DataFrame['_VolumeM'], color='green')
    ax3.legend(line3, ('%s'%DataFrame.columns[6],))
plt.show()