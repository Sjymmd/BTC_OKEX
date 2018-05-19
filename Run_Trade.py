from sklearn import preprocessing
from Class_Get_Data import *
from Class_Trade import *
from Model_DQN import *
from keras.models import load_model
import pickle

warnings.filterwarnings("ignore")

Get_Data = Get_Data()
Coin = Get_Data.Coin
Trade_api = Trade_Api()
names = locals()

skiprows = 240
k = 2

class Trade():

    def __init__(self,Price_Begun, action_last=len(Coin)):

        self.ValueAccount = action_last
        self.Price_Begun = Price_Begun
        self.Trade_Sign = 1
        self.Trade_Sign_Pre = 1
        self.ProfitLoss = 1
        self.skiprows = skiprows
        self.k = k

    def main(self):

        true = ''
        false = ''

        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        CoinName = Coin[self.ValueAccount] if self.ValueAccount != len(Coin) else 'CNY'

        #Trade
        if Trade_api.Check_FreezedCoin():
            order_id = eval(okcoinSpot.orderinfo(CoinName, -1))['orders']
            if order_id:
                order_id = order_id[0]['order_id']
                okcoinSpot.cancelOrder(CoinName, order_id)
                sellprice = float(okcoinSpot.ticker(Coin[self.ValueAccount])['ticker'][
                                  'buy']) if self.ValueAccount != len(Coin) else 1
                Trade_api.Sell_Coin()
                SellPrice = sellprice * Get_Data._USDT_CNY
                Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998
                names['QTY%s' % self.ValueAccount] = 0

                while True:
                    if Trade_api.Check_FreezedCoin():
                        time.sleep(10)
                    else:
                        print('Initial_Sell_Compete')
                        break

                f = open(Trade_Path, 'r+')
                f.read()
                f.write('\n%s' % now)
                f.write('\nSell %s , Price %s, Current_Profit %s' % (
                    CoinName, SellPrice, Cny - Initial_Asset))
                f.write('\nBuy %s , Price %s' % ('CNY',Get_Data._USDT_CNY ))
                f.close()

                self.ValueAccount = len(Coin)
                names['QTY%s' % self.ValueAccount] = Cny / Get_Data._USDT_CNY
        #Trade

        agent = DQN(self_print=True)

        Get_Data.GetData_Now()
        scaler = preprocessing.StandardScaler()

        with open('./Data/Data.pickle', 'rb') as myfile:
            Data = pickle.load(myfile)
        Data = Data[-self.skiprows:,:]
        # Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)[-self.skiprows:,:]
        Data = scaler.fit_transform(Data)

        # PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)[-self.skiprows:,:]
        with open('./Data/PriceArray.pickle', 'rb') as myfile:
            PriceArray = pickle.load(myfile)
        PriceArray = PriceArray[-self.skiprows:,:]
        number = -1
        state = Data[number,:]
        action = agent.action(state)

        Price_Now = PriceArray[:, action]
        gamma = 0.95
        fex = 1 / (0.998 * 0.998) - 1
        f_reward = 0
        for x in range(0, 2):
            f_reward += gamma * (((Price_Now[number - x] - Price_Now[number - x - 1]) / Price_Now[number - x]) - fex)
            gamma = gamma ** 2
        action_reward = float(f_reward)
        d_price = max(Price_Now) - Price_Now[number]

        insert = np.array([action_reward, agent.Q_Value, d_price])
        insert = scaler.fit_transform(insert.reshape((-1, 1))).reshape(insert.shape[0], )
        ClassifierDa = (state.tolist() + insert.tolist()+[action])
        ClassifierDa = np.array([ClassifierDa])

        ClassifierModel = load_model('./Keras_Model/my_model_classifier%d.h5'%self.k)

        Pre = ClassifierModel.predict_classes(ClassifierDa, verbose=0)

        tf.reset_default_graph()

        # Price = PriceArray[number, action]
        # SellPrice = PriceArray[number, self.ValueAccount]
        buyprice = float(okcoinSpot.ticker(Coin[action])['ticker']['buy'])if action != len(Coin) else 1
        Price = buyprice*Get_Data._USDT_CNY
        sellprice = float(okcoinSpot.ticker(Coin[self.ValueAccount])['ticker']['sell'])if self.ValueAccount != len(Coin) else 1
        SellPrice = sellprice*Get_Data._USDT_CNY

        if self.Trade_Sign_Pre == 0:
            self.Trade_Sign = 0

        # if Pre < self.k-1:
        #     action = self.ValueAccount

        if (SellPrice - self.Price_Begun) / self.Price_Begun > 0.1:
            # action = len(Coin)
            # Price = PriceArray[number, -1]
            # print('Profit Get!')
            # if self.Trade_Sign == 0:
            #     self.Trade_Sign = 1
            #     self.Trade_Sign_Pre = 1
            pass

        if (SellPrice - self.Price_Begun) / self.Price_Begun < -0.1:
            # action = len(Coin)
            # Price = PriceArray[number, action]
            # print('Profit Loss!')
            # self.Trade_Sign_Pre = 0
            # self.ProfitLoss = 0
            pass

        if action != self.ValueAccount:

            if self.Trade_Sign == 1:

                if self.Trade_Sign_Pre == 1 and self.ProfitLoss == 0:
                    action = len(Coin)
                    Price = PriceArray[number, -1]
                    self.ProfitLoss = 1

                else:

                    BUY_COIN = Coin[action] if action != len(Coin) else 'CNY'
                    SELL_COIN = Coin[self.ValueAccount] if self.ValueAccount != len(Coin) else 'CNY'

                    now = datetime.datetime.now()
                    now = now.strftime('%Y-%m-%d %H:%M:%S')


                    #Trade
                    Trade_api.Get_Coin()
                    Trade_api.Sell(SELL_COIN,sellprice)
                    while True:

                        time.sleep(10)
                        if Trade_api.Check_FreezedCoin():
                            FreezeCoin = Trade_api.Check_FreezedCoin()[0]
                            order_id = eval(okcoinSpot.orderinfo(FreezeCoin, -1))['orders']
                            if order_id:
                                order_id = order_id[0]['order_id']
                                okcoinSpot.cancelOrder(FreezeCoin, order_id)
                                sellprice = float(okcoinSpot.ticker(Coin[self.ValueAccount])['ticker'][
                                                  'buy']) if self.ValueAccount != len(Coin) else 1
                                Trade_api.Sell(SELL_COIN, sellprice)
                        else:
                            print('Sell Complete')
                            break
                    #Trade


                    SellPrice = sellprice * Get_Data._USDT_CNY
                    Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998
                    names['QTY%s' % self.ValueAccount] = 0

                    buyprice = float(okcoinSpot.ticker(Coin[action])['ticker']['buy']) if action != len(Coin) else 1
                    Price = buyprice * Get_Data._USDT_CNY

                    print('Time', now, 'Sell %s' % SELL_COIN, 'Buy %s' % BUY_COIN, 'Price', Price)
                    print('Sell %s' % SELL_COIN, 'Price', SellPrice, 'Current_Profit', Cny - Initial_Asset)

                    names['QTY%s' % action] = (Cny / Price) * 0.998
                    print('Buy %s' % BUY_COIN, 'Price', Price, 'Time', now)

                    # Trade
                    Trade_api.Get_Coin()
                    Trade_api.Buy(BUY_COIN,buyprice)
                    #Trade

                    #while True:
                    #     time.sleep(10)
                    #     if Trade_api.Check_FreezedCoin():
                    #         order_id = eval(okcoinSpot.orderinfo(BUY_COIN, -1))['orders']
                    #         if order_id:
                    #             order_id = order_id[0]['order_id']
                    #             okcoinSpot.cancelOrder(BUY_COIN, order_id)
                    #             buyprice = float(okcoinSpot.ticker(BUY_COIN)['ticker']['sell']) if action != len(
                    #                 Coin) else 1
                    #             Price = buyprice * Get_Data._USDT_CNY
                    #             Trade_api.Buy(BUY_COIN, buyprice)
                    #     else:
                    #         print('Buy Complete')
                    #         break


                    f = open(Trade_Path, 'r+')
                    f.read()
                    f.write('\n%s' % now)
                    f.write('\nSell %s , Price %s, Current_Profit %s' % (
                        SELL_COIN, SellPrice, Cny - Initial_Asset))
                    f.write('\nBuy %s , Price %s' % (BUY_COIN, Price))
                    f.close()

            self.Price_Begun = Price
            self.ValueAccount = action

        # CoinPrice = 0
        # for y in range(len(Coin)):
        #     CoinPrice += PriceArray[number,y] * names['QTY%s' % y]
        #     if names['QTY%s' % y] > 0:
        #         CoinName = Coin[y] if y !=len(Coin) else 'CNY'
        #         print('%s QTY' % CoinName, names['QTY%s' % y], ' Last_Price %s' % PriceArray[number,y])
        #         break

        CoinName = Coin[self.ValueAccount] if self.ValueAccount != len(Coin) else 'CNY'
        LastPrice = float(okcoinSpot.ticker(Coin[self.ValueAccount])['ticker']['last'])*Get_Data._USDT_CNY if self.ValueAccount != len(Coin) else Get_Data._USDT_CNY
        print('%s QTY' % CoinName, names['QTY%s' % self.ValueAccount], ' Last_Price %s' % LastPrice)
        CoinPrice = LastPrice*names['QTY%s' % self.ValueAccount]
        profit = CoinPrice - Initial_Asset

        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now, 'Profit:%d' % profit, 'Total Asset:%d' % (profit + Initial_Asset))

        Asset = Trade_api.GetAsset()
        print('Actual_Asset',Asset)

        # print('Value','Pre',Pre,'self.Trade_Sign',self.Trade_Sign,'self.ProfitLoss',self.ProfitLoss,'self.action_last',self.action_last )

if __name__ == '__main__':

    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')

    try:
        Trade_Path = './Log/Trade_Log.txt'
        f = open(Trade_Path, 'r+')
        ValueAccount_Txt = f.readlines()
        # f.read()
        # f.write('CreateTime %s' % now)
        f.close()
        Initial_Asset = float(str(ValueAccount_Txt[0]).split(' ')[-1])
        Current_Profit = float(str(ValueAccount_Txt[-2]).split(' ')[-1][:-1])
        Price_Begun = float(str(ValueAccount_Txt[-1]).split(' ')[-1])
        Total_Asset = Initial_Asset + Current_Profit
        ValueAccount = str(ValueAccount_Txt[-1]).split(' ')[1]
        QTY = float(Total_Asset / float(str(ValueAccount_Txt[-1]).split(' ')[-1]))
        print('Successfully loaded:Trade_Log')

    except:

        ValueAccount = 'CNY'
        from Class_Trade import Trade_api
        Trade_api = Trade_api()
        Total_Asset = Trade_api.GetAsset()
        QTY = float(Total_Asset / Get_Data._USDT_CNY)
        Price_Begun = Get_Data._USDT_CNY
        print("Initial Model")

    try:
        action_last = Coin.index(ValueAccount)
    except:
        action_last = len(Coin)
    names = locals()
    for x in range(len(Coin)+1):
        names['QTY%s' % x] = 0

    names['QTY%s' % action_last] = QTY

    Trade = Trade(action_last=action_last, Price_Begun=Price_Begun)

    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler()

    def job():
        Trade.main()

    while True:

        sched.add_job(job, 'interval', hours=4)
        # sched.add_job(job, 'cron', minute=1)

        try:
            sched.start()

        except:
            print('定时任务出错')
            time.sleep(10)
            continue