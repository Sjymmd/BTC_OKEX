# coding: utf-8
from DQN_Model import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from Volume_Early_Warning import *
Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24 * 1000
Okex_Api._EndLenth = 0

def Get_Dataframe(Coin):

    try:
        DataFrame = pd.DataFrame(
            columns=("Coin", "Cny", "High", "Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
        data = pd.DataFrame(
            okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChosen], Okex_Api._Lenth, Okex_Api._EndLenth, Coin)).iloc[
               :Okex_Api._Lenth - 1, ]
        data[5] = data.iloc[:, 5].apply(pd.to_numeric)
        data = data[data[5] >= 1000]
        data = data.reset_index(drop=True)
        Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
        # Increase = str('%.2f' % (Increase) + '%')
        price = float(data.iloc[0, 4])
        Hi_price = round(float((data.iloc[0, 2])) * Okex_Api._USDT_CNY, 2)
        Lo_price = round(float((data.iloc[0, 3])) * Okex_Api._USDT_CNY, 2)
        Cny = round(price * Okex_Api._USDT_CNY, 2)
        Volume = float(data.iloc[0, 5])
        Volume_Mean = round(Volume / 1000, 2)
        Volume_Pre = round(Volume / 1000, 2)
        Volume_Pre_P = 0
        if Volume_Mean == 0:
            Volume_Inc = 0
        else:
            Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
        Timeshrft = pd.Series(
            {'Coin': Coin, 'Cny': Cny, 'High': Hi_price, 'Low': Lo_price, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
             'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        for lenth in range(1, Okex_Api._Lenth - 1):
            try:
                Increase = (float(data.iloc[lenth, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
                # Increase = str('%.2f' % (Increase) + '%')
                price = float(data.iloc[lenth, 4])
                Hi_price = round(float((data.iloc[lenth, 2])) * Okex_Api._USDT_CNY, 2)
                Lo_price = round(float((data.iloc[lenth, 3])) * Okex_Api._USDT_CNY, 2)
                Cny = round(price * Okex_Api._USDT_CNY, 2)
                Volume = data.iloc[:lenth + 1, 5].apply(pd.to_numeric)
                Volume_Mean = round(Volume.mean() / 1000, 2)
                Volume_Pre = round(Volume.iloc[lenth] / 1000, 2)
                Volume_Pre_P = round((Volume[lenth] / Volume[lenth - 1]) - 1, 2)
                Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
                Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny, 'High': Hi_price, 'Low': Lo_price, 'Inc': Increase,
                                       'Volume_Pre_K': Volume_Pre,
                                       'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
                DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
            except:
                break
        return DataFrame
        # print(DataFrame)
    except:
        time.sleep(5)
        print('%sError' % Coin)


class DQN_Select:
    def __init__(self):
        self.Coin = Coin

    def DQN_Select(self):
        StartTime = time.time()
        Count = 0
        Profit_Total = 0
        DataFrame = pd.DataFrame(
            columns=("Coin", "Profit"))
        for x in self.Coin:
            try:
                scaler = preprocessing.StandardScaler()
                scaler_Price = preprocessing.StandardScaler()
                TestData = Get_Dataframe(x)
                if len(TestData) < 1000:
                    print('%s less than 1000 lines' % x)
                    continue
                else:
                    Count += 1
                TestData = TestData.iloc[:, 1:]

                TestPrice = TestData.iloc[:, 0]
                TestPrice = TestPrice.reshape(-1, 1)
                TestPrice = scaler_Price.fit_transform(TestPrice)

                TestData_Initial = TestData.as_matrix()
                TestData = scaler.fit_transform(TestData_Initial)
                tf.reset_default_graph()
                env1 = TWStock(TestData)
                state = env1.reset()
                agent = DQN(env1)
                Cny = 1000
                Coin = 0
                for i in range(len(TestData) - 1):
                    env1.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env1.step(action)
                    Price = scaler_Price.inverse_transform(state[0].reshape(-1, 1))
                    Price = round(Price[0][0], 2)
                    if action == 1 and done is False and Cny > 0:
                        # print('Buy','Time',i,'Price',Price)
                        Coin = Cny / Price
                        Cny = 0
                    elif action == 2 and Coin > 0 and done is False:
                        Cny = Coin * Price
                        Coin = 0
                        # print('Sell','Time',i,'Price', Price,'Current_Profit',Cny +Coin*Price-1000)
                profit = Cny + Coin * Price - 1000
                Timeshrft = pd.Series({'Coin': x, 'Profit': profit})
                DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
                print('%s Profit:%d' % (x, profit))
            except:
                continue
            Profit_Total += profit
            AvgProfit = profit / Count
        DataFrame = DataFrame.sort_values(by='Profit', ascending=False)
        DataFrame = DataFrame.reset_index(drop=True)
        EndTime = time.time()
        print('Using_Time: %d min' % int((EndTime - StartTime) / 60))
        print('TotalProfit', Profit_Total, 'AvgProfit', AvgProfit)
        # print(DataFrame.iloc[:5,0].values)
        DataFrame.to_csv('Coin_Select.txt', index=False)

if __name__=='__main__':
    DQN_Select = DQN_Select()
    DQN_Select.DQN_Select()