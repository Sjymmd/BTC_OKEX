from sklearn import preprocessing
from Model_DQN_Increase import *
from Trade import *

from keras.models import load_model

warnings.filterwarnings("ignore")

# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)

Okex_Api = Okex_Api()
Okex_Api._Lenth = 24*100
Coin = Okex_Api.GetCoin()

names = locals()
StartTime = time.time()

class Trade():

    def __init__(self, Price_Begun, action_last=len(Coin), ValuAccount='CNY'):

        self.action_last = action_last
        self.ValueAccount = ValuAccount
        self.Price_Begun = Price_Begun
        self.Trade_Sign = 1
        self.Trade_Sign_Pre = 1
        self.ProfitLoss = 1

    def main(self):

        # print('Start Loading Data...')
        # StartTime = time.time()

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
        print('MinLenth', lenData)
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

        number = -1

        agent = DQN(self_print=False)
        state = Data[number]

        action = agent.action(state)
        self.action_last = action

        CoinName = Coin[self.action_last] if self.action_last < len(Coin) else 'CNY'
        DataNow = names['TestPrice%s' % CoinName]
        # print(DataNow)
        gamma = 0.95
        fex = 1 / (0.998 * 0.998) - 1
        f_reward = 0
        for x in range(0, 2):
            f_reward += gamma * (((DataNow[number - x] - DataNow[number - x - 1]) / DataNow[number - x]) - fex)
            gamma = gamma ** 2
        action_reward = float(f_reward)
        d_price = max(DataNow) - DataNow[number]

        insert = np.array([action_reward, agent.Q_Value, action, d_price])

        ClassifierDa = (state.tolist() + insert.tolist())
        ClassifierDa = np.array([ClassifierDa])

        ClassifierModel = load_model('./Keras_Model/my_model.h5')

        Pre = ClassifierModel.predict_classes(ClassifierDa, verbose=0)

        tf.reset_default_graph()

        Price = [USDT_CNY] if CoinName == 'CNY' else names['TestPrice%s' % CoinName][number]
        Price = Price[0]
        SellPrice = [USDT_CNY] if self.ValueAccount == 'CNY' else names['TestPrice%s' % self.ValueAccount][number]
        SellPrice = SellPrice[0]

        if self.Trade_Sign_Pre == 0:
            self.Trade_Sign = 0

        if Pre < 1:
            CoinName = self.ValueAccount

        if (SellPrice - self.Price_Begun) / self.Price_Begun > 0.1:
            CoinName = 'CNY'
            Price = [USDT_CNY][0]
            print('Profit Get!')
            if self.Trade_Sign == 0:
                self.Trade_Sign = 1
                self.Trade_Sign_Pre = 1

        if (SellPrice - self.Price_Begun) / self.Price_Begun < -0.1:
            CoinName = 'CNY'
            Price = [USDT_CNY][0]
            print('Profit Loss!')
            self.Trade_Sign_Pre = 0
            self.ProfitLoss = 0

        if CoinName != self.ValueAccount:

            if self.Trade_Sign == 1:

                if self.Trade_Sign_Pre == 1 and self.ProfitLoss == 0:
                    CoinName = 'CNY'
                    Price = [USDT_CNY][0]
                    self.ProfitLoss = 1

                else:

                    Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998
                    names['QTY%s' % self.ValueAccount] = 0

                    now = datetime.datetime.now()
                    now = now.strftime('%Y-%m-%d %H:%M:%S')

                    print('Time', number, 'Sell %s' % self.ValueAccount, 'Buy %s' % CoinName, 'Price', Price)
                    print('Sell %s' % self.ValueAccount, 'Price', SellPrice, 'Current_Profit', Cny - Initial_Asset)

                    names['QTY%s' % CoinName] = (Cny / Price) * 0.998

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

                    f = open(Trade_Path, 'r+')
                    f.read()
                    f.write('\n%s' % now)
                    f.write('\nSell %s , Price %s, Current_Profit %s' % (
                        self.ValueAccount, SellPrice, Cny - Initial_Asset))
                    f.write('\nBuy %s , Price %s' % (CoinName, Price))
                    f.close()

            self.Price_Begun = Price
            self.ValueAccount = CoinName

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

        Asset = Trade_api.GetAsset()
        print('Actual_Asset',Asset)

if __name__ == '__main__':

    Coin = np.loadtxt("./logs/Coin_Select.txt", dtype=np.str)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    Trade_api = Trade_Api()

    while True:
        try:
            USDT_CNY = okcoinfuture.exchange_rate()['rate']
            Initial_Asset = Trade_api.GetAsset()
            break
        except:
            # print('Get_Dataframe Error')
            time.sleep(10)
            continue

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

    def job():
        Trade.main()

    while True:

        # sched.add_job(job, 'interval', seconds=30)
        sched.add_job(job, 'cron', minute=1)

        try:
            sched.start()

        except:
            print('定时任务出错')
            time.sleep(10)
            continue