from Volume_Early_Warning import *
from sklearn import preprocessing
from Model_DQN_Increase import *

warnings.filterwarnings("ignore")
Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24 * 1000
Okex_Api._EndLenth = 0
# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)
names = locals()
StartTime = time.time()


def Get_Dataframe(Coin):
    try:
        DataFrame = pd.DataFrame(
            columns=("Coin", "Cny", "High", "Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
        data = pd.DataFrame(
            okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChosen], Okex_Api._Lenth, Okex_Api._EndLenth,
                                Coin)).iloc[:Okex_Api._Lenth - 1, ]
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
        for lenth in range(1, len(data) - 1):
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
        time.sleep(10)
        print('%s error' % Coin)


class Trade():

    def __init__(self, Price_Begun, action_last=len(Coin), ValuAccount='CNY'):

        self.action_last = action_last
        self.ValueAccount = ValuAccount
        self.Price_Begun = Price_Begun

    def main(self):
        # print('Start Loading Data...')
        # StartTime = time.time()

        CoinName = Coin[self.action_last] if self.action_last < len(Coin) else 'CNY'
        Chosen = [CoinName,self.ValueAccount]

        for x in Chosen:

            if x =='CNY':
                names['TestPriceCNY'] = pd.DataFrame(columns=['A'])
                for x in range(10):
                    names['TestPriceCNY'] = names['TestPriceCNY'].append({'A': USDT_CNY}, ignore_index=True)
                names['TestPrice%s' % 'CNY'] = names['TestPrice%s' % 'CNY']['A'].reshape(-1, 1)
            else:
                while True:
                    try:
                        TestData = Get_Dataframe(x)
                    except:
                        # print('Get_Dataframe Error')
                        time.sleep(10)
                        continue
                    if TestData is not None:
                        break
                    # print('Get %s error' % x)
                TestData = TestData.iloc[:, 1:]
                names['TestPrice%s' % x] = TestData.iloc[:, 0]
                names['TestPrice%s' % x] = names['TestPrice%s' % x].reshape(-1, 1)

        DataNow = names['TestPrice%s' % CoinName]

        gamma = 0.95
        fex = 1 / (0.998 * 0.998) - 1
        f_reward = 0
        for x in range(1, 3):
            # print(len(NowData))
            f_reward += gamma * (((DataNow[-x] - DataNow[-x - 1]) / DataNow[-x]) - fex)
            gamma = gamma ** 2
        # action_reward = (NowData*gamma + f_reward)*count

        action_reward = float(f_reward)
        # print('Action_reward_last',action_reward,'Action_last',self.action_last)
        if action_reward < 0.01 and CoinName != 'CNY':
            CoinName = self.ValueAccount

        Price = [USDT_CNY] if CoinName == 'CNY' else names['TestPrice%s' % CoinName][-1]
        Price = round(Price[0], 2)
        SellPrice = [USDT_CNY] if self.ValueAccount == 'CNY' else names['TestPrice%s' % self.ValueAccount][-1]
        SellPrice = round(SellPrice[0], 2)

        if (SellPrice - self.Price_Begun) / self.Price_Begun > 0.1:
            CoinName = 'CNY'
            Price = round([USDT_CNY][0], 2)
            print('Profit Get!')

        if CoinName != self.ValueAccount:
            Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998

            names['QTY%s' % self.ValueAccount] = 0

            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')

            # print('\033[32;0mSell %s\033[0m' % self.ValueAccount, 'Price', SellPrice, 'Current_Profit', Cny - Initial_Asset)
            # print('\033[31;0mBuy %s\033[0m' % CoinName, 'Price', Price, 'Time', now)
            print('Sell %s' % self.ValueAccount, 'Buy %s' % CoinName)
            print('Sell %s' % self.ValueAccount, 'Price', SellPrice, 'Current_Profit', Cny - Initial_Asset)

            # Trade_api.Get_Coin()
            # Trade_api.Sell_Coin()
            #
            #
            # while True:
            #    if Trade_api.Check_FreezedCoin():
            #        time.sleep(5)
            #    else:
            #        print('Sell Complete')
            #        break
            #
            # print('Buy %s' % CoinName, 'Price', Price, 'Time', now)
            #
            # Trade_api.Get_Coin()
            # Trade_api.Buy_Coin(CoinName)


            self.Price_Begun = Price
            f = open(Trade_Path, 'r+')
            f.read()
            f.write('\n%s' % now)
            f.write('\nSell %s , Price %s, Current_Profit %s' % (
                self.ValueAccount, SellPrice, Cny - Initial_Asset))
            f.write('\nBuy %s , Price %s' % (CoinName, Price))
            f.close()

            names['QTY%s' % CoinName] = (Cny / Price) * 0.998
            self.ValueAccount = CoinName

        DataLen = []
        for x in Coin:
            scaler = preprocessing.StandardScaler()
            while True:
                try:
                    TestData = Get_Dataframe(x)
                except:
                    # print('Get_Dataframe Error')
                    time.sleep(10)
                    continue
                if TestData is not None:
                    break
                # print('Get %s error' % x)
            TestData = TestData.iloc[:, 1:]
            TestData_Initial = TestData.as_matrix()
            names['TestPrice%s' % x] = TestData.iloc[:, 0]
            names['TestPrice%s' % x] = names['TestPrice%s' % x].reshape(-1, 1)
            names['TestData%s' % x] = scaler.fit_transform(TestData_Initial)
            DataLen.append(names['TestData%s' % x].shape[0])

        lenData = min(DataLen)
        names['TestPriceCNY'] = pd.DataFrame(columns=['A'])
        for x in range(lenData):
            names['TestPriceCNY'] = names['TestPriceCNY'].append({'A': USDT_CNY}, ignore_index=True)
        names['TestPrice%s' % 'CNY'] = names['TestPrice%s' % 'CNY']['A'].reshape(-1, 1)

        # print(names['TestPriceCNY'])
        # print('MinLenth', lenData)
        Tem = names['TestData%s' % Coin[0]]
        Tem_Price = names['TestPrice%s' % Coin[0]]
        names['TestPrice%s' % Coin[0]] = Tem_Price[int(len(Tem) - lenData):]
        names['TestData%s' % Coin[0]] = Tem[int(len(Tem) - lenData):]
        Data = names['TestData%s' % Coin[0]]
        for x in Coin[1:]:
            Tem = names['TestData%s' % x]
            Tem_Price = names['TestPrice%s' % x]
            names['TestPrice%s' % x] = Tem_Price[int(len(Tem) - lenData):]
            names['TestData%s' % x] = Tem[int(len(Tem) - lenData):]
            Data = np.column_stack((Data, names['TestData%s' % x]))

        # EndTime = time.time()
        # print('Loading Data Using_Time: %d min' % int((EndTime - StartTime) / 60))

        DataNow = Data[-1]

        env1 = TWStock(DataNow)
        state = DataNow
        agent = DQN(env1, self_print=False)
        agent.print = False

        # env1.render()
        action = agent.action(state)  # direct
        tf.reset_default_graph()
        self.action_last = action

        CoinPrice = 0
        for x in Coin:

            CoinPrice += names['TestPrice%s' % x][-1] * names['QTY%s' % x]
            if names['QTY%s' % x] > 0:
                print('%s QTY' % x, names['QTY%s' % x], ' Last_Price %s' % names['TestPrice%s' % x][-1][0])
                break

        if names['QTYCNY'] > 0:
            CoinPrice += USDT_CNY * names['QTYCNY']
            print('CNY QTY', names['QTYCNY'], ' Last_Price %s' % USDT_CNY)
        profit = CoinPrice - Initial_Asset

        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now, 'Profit:%d' % profit, 'Total Asset:%d' % (profit + Initial_Asset),'Rewards',action_reward)


if __name__ == '__main__':

    Coin = np.loadtxt("Coin_Select.txt", dtype=np.str)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    USDT_CNY = okcoinfuture.exchange_rate()['rate']
    Initial_Asset = 1000
    try:
        Trade_Path = 'Trade_Log.txt'
        f = open(Trade_Path, 'r+')
        ValueAccount_Txt = f.readlines()
        # f.read()
        # f.write('CreateTime %s' % now)
        f.close()

        Current_Profit = float(str(ValueAccount_Txt[-2]).split(' ')[-1][:-1])
        Price_Begun = float(str(ValueAccount_Txt[-1]).split(' ')[-1])
        Total_Asset = Initial_Asset + Current_Profit
        ValueAccount = str(ValueAccount_Txt[-1]).split(' ')[1]
        QTY = float(Total_Asset / float(str(ValueAccount_Txt[-1]).split(' ')[-1])) * 0.998
        print('Successfully loaded:Trade_Log')

    except:

        ValueAccount = 'CNY'
        Total_Asset = Initial_Asset
        QTY = float(Total_Asset / USDT_CNY)
        Price_Begun = USDT_CNY
        print("Initial Model")

    try:
        action_last = Coin.tolist().index(ValueAccount)
    except:
        action_last = len(Coin)

    names = locals()
    for x in Coin:
        names['QTY%s' % x] = 0
    names['QTYCNY'] = 0

    names['QTY%s' % ValueAccount] = QTY

    Trade = Trade(ValuAccount=ValueAccount, action_last=action_last, Price_Begun=Price_Begun)

    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler()

    def job():
        Trade.main()


    while True:

        sched.add_job(job, 'interval', seconds=30)
        # sched.add_job(job, 'cron', minute=2)

        try:
            sched.start()

        except:
            print('定时任务出错')
            time.sleep(10)
            continue

