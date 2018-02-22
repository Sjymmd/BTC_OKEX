from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24*100
Okex_Api._EndLenth = 0
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')
print(now)
StartTime = time.time()
for x in Coin[:int(Okex_Api._CoinLenth)]:
    try:
        DataFrame = pd.DataFrame(columns=("Coin", "Cny","High","Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
        data = pd.DataFrame(okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChosen], Okex_Api._Lenth, Okex_Api._EndLenth, x)).iloc[:Okex_Api._Lenth-1, ]
        data[5] = data.iloc[:, 5].apply(pd.to_numeric)
        data = data[data[5] >= 1000]
        data = data.reset_index(drop=True)
        Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
        price = float(data.iloc[0, 4])
        Hi_price = round(float((data.iloc[0, 2]))* Okex_Api._USDT_CNY,2)
        Lo_price = round(float((data.iloc[0, 3]))* Okex_Api._USDT_CNY,2)
        Cny = round(price * Okex_Api._USDT_CNY, 2)
        Volume = float(data.iloc[0, 5])
        Volume_Mean = Volume/1000
        Volume_Pre = round(Volume / 1000, 2)
        Volume_Pre_P = 0
        if Volume_Mean == 0:
            Volume_Inc = 0
        else:
            Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
        Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny,'High':Hi_price,'Low':Lo_price, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                               'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        for lenth in range(1,Okex_Api._Lenth-1):
            try:
                Increase = (float(data.iloc[lenth, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
                Increase = str('%.2f' % (Increase) + '%')
                price = float(data.iloc[lenth, 4])
                Hi_price = round(float((data.iloc[lenth, 2])) * Okex_Api._USDT_CNY, 2)
                Lo_price = round(float((data.iloc[lenth, 3])) * Okex_Api._USDT_CNY, 2)
                Cny = round(price * Okex_Api._USDT_CNY, 2)
                Volume = data.iloc[:lenth+1, 5].apply(pd.to_numeric)
                Volume_Mean = round(Volume.mean() / 1000, 2)
                Volume_Pre = round(Volume.iloc[lenth] / 1000, 2)
                Volume_Pre_P = round((Volume[lenth] / Volume[lenth - 1])-1, 2)
                Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
                Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny,'High':Hi_price,'Low':Lo_price, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                                       'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
                DataFrame = DataFrame.append(Timeshrft, ignore_index=True)

            except:
                break

        # print(DataFrame)
        Coin_Snt= plt.figure('%s'%x,figsize=(12,6 ))
        plt.grid()
        ax1 = plt.subplot(411)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.xaxis.grid(True, which='major')
        ax1.yaxis.grid(True, which='major')
        line1=ax1.plot(DataFrame['Cny'], linewidth=1.0, color='blue')
        ax1.legend(line1, ('%s-%d %s'%(x,len(DataFrame['Cny'])+1,str(Okex_Api._KlineChosen)),))

        ax2 = plt.subplot(412)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.xaxis.grid(True, which='major')
        ax2.yaxis.grid(True, which='major')
        line2 = ax2.bar(np.arange(len(DataFrame['Volume_Pre_K'])),DataFrame['Volume_Pre_K'], color='red')
        line_ = ax2.plot(DataFrame['Mean_Volume_K'],color='purple')
        ax2.legend(line2, ('%s' % DataFrame.columns[5],))
        # ax2.legend(line22, ('%s' % DataFrame.columns[4],))

        ax3 = plt.subplot(413)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.xaxis.grid(True, which='major')
        ax3.yaxis.grid(True, which='major')
        line3=ax3.bar(np.arange(len(DataFrame['_VolumeS'])),DataFrame['_VolumeS'], color='orange')
        ax3.legend(line3, ('%s'%DataFrame.columns[7],))

        ax4= plt.subplot(414)
        ax4.xaxis.grid(True, which='major')
        ax4.yaxis.grid(True, which='major')
        line4 = ax4.bar(np.arange(-Okex_Api._EndLenth,len(DataFrame['_VolumeM'])-Okex_Api._EndLenth),DataFrame['_VolumeM'], color='green')
        ax4.legend(line4, ('%s'%DataFrame.columns[8],))
        Coin_Snt.savefig("./Charts/%s.png"%x)
        # plt.show()
    except:
        print('%sError'%x)
        continue

print('Charts Success')
EndTime = time.time()
print('Using_Time: %d min'%int((EndTime - StartTime)/60))