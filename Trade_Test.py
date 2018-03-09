from Train_DQN_Increase import Get_Dataframe
from Volume_Early_Warning import *
from sklearn import preprocessing
from Model_DQN_Increase import *
warnings.filterwarnings("ignore")

class Trade():

    def __init__(self):

        self.action_last = len(Coin)
        self.ValueAccount = 'CNY'


    def main(self):
        # print('Start Loading Data...')
        # StartTime = time.time()

        DataLen = []
        for x in Coin:
            scaler = preprocessing.StandardScaler()
            while True:
                try:
                    TestData = Get_Dataframe(x)
                except:
                    # print('Get_Dataframe Error')
                    time.sleep(5)
                    continue
                if TestData is not None:
                    break
                # print('Get %s error' % x)
            TestData = TestData.iloc[:-1, 1:]
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

        EndTime = time.time()
        # print('Loading Data Using_Time: %d min' % int((EndTime - StartTime) / 60))

        DataNow = Data[-1]

        env1 = TWStock(DataNow)
        state = DataNow
        agent = DQN(env1)

        # env1.render()
        action = agent.action(state)  # direc

        CoinName = Coin[self.action_last] if self.action_last < len(Coin) else 'CNY'

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

        if CoinName != self.ValueAccount:

            Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998

            names['QTY%s' % self.ValueAccount] = 0

            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')

            print('Sell %s' % self.ValueAccount, 'Price', SellPrice, 'Current_Profit', Cny - Total_Asset)
            print('Buy %s' % CoinName, 'Price', Price, 'Time', now)

            f = open(Trade_Path, 'r+')
            f.read()
            f.write('\n%s'%now)
            f.write('\nSell %s , Price %s, Current_Profit %s' % (
                self.ValueAccount, SellPrice, Cny - Total_Asset))
            f.write('\nBuy %s , Price %s' % (CoinName, Price))

            names['QTY%s' % CoinName] = (Cny / Price) * 0.998
            self.ValueAccount = CoinName


        tf.reset_default_graph()
        self.action_last = action

        CoinPrice = 0
        for x in Coin:
            CoinPrice += names['TestPrice%s' % x][-1] * names['QTY%s' % x]
            if names['QTY%s' % x] > 0:
                print('%s QTY' % x, names['QTY%s' % x], ' Last_Price %s' % names['TestPrice%s' % x][-1][0])
        if names['QTYCNY'] > 0:
            CoinPrice += USDT_CNY * names['QTYCNY']
            print('QTYCNY', names['QTYCNY'], ' Last_Price %s' % USDT_CNY)
        profit = CoinPrice - Total_Asset

        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now, 'Profit:%d' % profit, 'Total Asset:%d' % (profit + Total_Asset))





if __name__=='__main__':

    Coin = np.loadtxt("Coin_Select.txt", dtype=np.str)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    USDT_CNY = okcoinfuture.exchange_rate()['rate']

    Trade_Path = 'Trade_Log.txt'
    with open(Trade_Path, "w") as f:
        f.write('CreateTime %s' % now)

    Total_Asset  = 1000

    names = locals()
    for x in Coin:
        names['QTY%s' % x] = 0
    names['QTYCNY'] = Total_Asset / round(USDT_CNY, 2)

    Trade = Trade()

    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler()

    def job():
        Trade.main()

    while True:

        # sched.add_job(job, 'interval', seconds=30)
        sched.add_job(job,'cron', minute = 2)

        try:
            sched.start()

        except:
            print('定时任务出错')
            time.sleep(10)
            continue

