from Volume_Early_Warning import *
from sklearn import preprocessing
from Model_DQN_Increase import *
from Trade import *

warnings.filterwarnings("ignore")

# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)

Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()

names = locals()
StartTime = time.time()

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

        if CoinName == 'CNY':
            Vol_Judge = True
        else:
            df = Okex_Api.GetDataCoin(CoinName)
            df2 = Okex_Api.GetDataCoin(CoinName,Clean=False)
            Vol_Judge = df.iloc[-1, 1] == df2.iloc[-1, 1]

        if Vol_Judge:

            for x in Chosen:

                if x =='CNY':
                    names['TestPriceCNY'] = pd.DataFrame(columns=['A'])
                    for x in range(10):
                        names['TestPriceCNY'] = names['TestPriceCNY'].append({'A': USDT_CNY}, ignore_index=True)
                    names['TestPrice%s' % 'CNY'] = names['TestPrice%s' % 'CNY']['A'].reshape(-1, 1)
                else:
                    while True:
                        try:
                            TestData = Okex_Api.GetDataCoin(x)
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

                Trade_api.Get_Coin()
                Trade_api.Sell_Coin()


                while True:
                   if Trade_api.Check_FreezedCoin():
                       time.sleep(5)
                   else:
                       print('Sell Complete')
                       break

                print('Buy %s' % CoinName, 'Price', Price, 'Time', now)

                Trade_api.Get_Coin()
                Trade_api.Buy_Coin(CoinName)


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
                        TestData = Okex_Api.GetDataCoin(x)
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
            print(now, 'Profit:%d' % profit, 'Total Asset:%d' % (profit + Initial_Asset))
        else:
            print('Volume_K<1K')

        Asset = Trade_api.GetAsset()
        print('Actual_Asset',Asset)

if __name__ == '__main__':

    Coin = np.loadtxt("./logs/Coin_Select.txt", dtype=np.str)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    USDT_CNY = okcoinfuture.exchange_rate()['rate']
    Initial_Asset = 1000
    try:
        Trade_Path = './logs/Trade_Log.txt'
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
    Trade_api = Trade_Api()

    def job():
        Trade.main()


    while True:

        # sched.add_job(job, 'interval', seconds=30)
        sched.add_job(job, 'cron', minute=2)

        try:
            sched.start()

        except:
            print('定时任务出错')
            time.sleep(10)
            continue

