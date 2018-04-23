from sklearn import preprocessing
from Class_Get_Data import *
from Model_DQN import *
from keras.models import load_model

warnings.filterwarnings("ignore")

# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)

Get_Data = Get_Data()
Coin = Get_Data.Coin

Read_Trade_Log = False

skiprows =1
lenth = -1

Max_Interval = 240
k =2



Trade_Path = './Log/Trade_Log.txt'

Path = './Keras_Model/my_model_classifier%d.h5'%k
Trade_Path_Testback = './Log/Trade_TestBack_classifier%d.txt'%k



class TestBack():

    def __init__(self, Price_Begun=Get_Data._USDT_CNY,action_last=len(Coin) ) :

        self.ValueAccount = action_last
        self.Price_Begun = Price_Begun
        self.Path = Path
        self.k =k
        self.skiprows = skiprows
        self.lenth = lenth
        self.GetData_DQN()
        # self.GetData_DDQN()

    def GetData_DQN(self):

        scaler = preprocessing.StandardScaler()
        # print(self.skiprows)
        Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)[-self.skiprows-self.lenth:-self.skiprows,:]
        Data = scaler.fit_transform(Data)
        PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)[-self.skiprows-self.lenth:-self.skiprows,:]

        DQN_Data = pd.DataFrame()
        agent = DQN()
        
        for number in range(2,len(Data)):

            state = Data[number,:]
            action = agent.action(state)
            Price_Now = PriceArray[:,action]
            gamma = 0.95
            fex = 1 / (0.998 * 0.998) - 1
            f_reward = 0
            for x in range( 0, 2):
                f_reward += gamma * (((Price_Now[number - x] - Price_Now[number - x - 1]) / Price_Now[number - x]) - fex)
                gamma = gamma ** 2
            action_reward = float(f_reward)

            d_price = max(Price_Now[number - Max_Interval  if (number-Max_Interval)>=0 else 0:number + 1]) - Price_Now[number]

            insert = np.array([action_reward, agent.Q_Value,d_price])
            insert = scaler.fit_transform(insert.reshape((-1, 1))).reshape(insert.shape[0], )
            ClassifierDa = (state.tolist() + insert.tolist()+[action])
            ClassifierDa = np.array([ClassifierDa])

            if number == 2:
                DQN_Data = ClassifierDa
            else:
                DQN_Data = np.row_stack((DQN_Data, ClassifierDa))

            self.DQN_Data = DQN_Data
            self.PriceArray = PriceArray[2:,:]

    def GetData_DDQN(self):

        from Train_DQN_Keras import DQNAgent

        scaler = preprocessing.StandardScaler()
        # print(self.skiprows)
        Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)[
               -self.skiprows - self.lenth:-self.skiprows, :]
        Data = scaler.fit_transform(Data)
        PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)[
                     -self.skiprows - self.lenth:-self.skiprows, :]

        state_size = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str)) * 9)
        action_size = int(state_size / 9 + 1)
        DQN_Data = pd.DataFrame()

        agent = DQNAgent(state_size=state_size, action_size=action_size)
        agent.load("./DQN_Model/Keras-ddqn.h5")
        print('Loading Keras-ddqn Model Successfully')

        for number in range(2,len(Data)):
            state = Data[number, :].reshape(1,-1)
            action = agent.action(state)
            state = Data[number, :].reshape(-1,)
            Price_Now = PriceArray[:, action]
            gamma = 0.95
            fex = 1 / (0.998 * 0.998) - 1
            f_reward = 0
            for x in range(0, 2):
                f_reward += gamma * (
                            ((Price_Now[number - x] - Price_Now[number - x - 1]) / Price_Now[number - x]) - fex)
                gamma = gamma ** 2
            action_reward = float(f_reward)

            d_price = max(Price_Now[number - Max_Interval if (number - Max_Interval) >= 0 else 0:number + 1]) - \
                      Price_Now[number]

            insert = np.array([action_reward, agent.Q_Value, d_price])
            insert = scaler.fit_transform(insert.reshape((-1, 1))).reshape(insert.shape[0], )
            ClassifierDa = (state.tolist() + insert.tolist()+ [action])
            ClassifierDa = np.array([ClassifierDa])

            if number == 2:
                DQN_Data = ClassifierDa
            else:
                DQN_Data = np.row_stack((DQN_Data, ClassifierDa))

        self.DQN_Data = DQN_Data
        self.PriceArray = PriceArray[2:, :]

    def Run_Testback(self):

        DQN_Data = self.DQN_Data
        ClassifierModel = load_model(self.Path)
        print('Successfully loaded : my_model_classifier%d'%self.k)
        f = open(Trade_Path_Testback, 'w+')
        f.truncate()
        f.close()

        Trade_Sign = 1
        Trade_Sign_Pre = 1
        ProfitLoss = 1

        for number in range(2,len(DQN_Data)):

            Classifier_Data = DQN_Data[number,:].reshape(1,DQN_Data[number,:].shape[0])
            Pre = ClassifierModel.predict_classes(Classifier_Data,verbose=0)

            Action = int(DQN_Data[number,-1])

            Price = self.PriceArray[number,Action]
            SellPrice = self.PriceArray[number, self.ValueAccount]

            if Trade_Sign_Pre == 0 :
                Trade_Sign =0

            # if Pre < self.k - 1 :
            #     Action = self.ValueAccount

            if (SellPrice - self.Price_Begun) / self.Price_Begun > 0.1:
                # Action = len(Coin)
                # Price = self.PriceArray[number,-1]
                # print('Profit Get!')
                # if Trade_Sign ==0:
                #     Trade_Sign =1
                #     Trade_Sign_Pre = 1
                pass

            if (SellPrice - self.Price_Begun) / self.Price_Begun < -0.1:
                # Action = len(Coin)
                # Price = self.PriceArray[number,Action]
                # print('Profit Loss!')
                # Trade_Sign_Pre = 0
                # ProfitLoss = 0
                pass

            if Action != self.ValueAccount:

                if Trade_Sign == 1:

                    if Trade_Sign_Pre == 1 and ProfitLoss== 0:
                        Action = len(Coin)
                        Price = self.PriceArray[number, -1]
                        ProfitLoss = 1

                    else:

                        Cny = names['QTY%s' % self.ValueAccount] * SellPrice * 0.998
                        names['QTY%s' % self.ValueAccount] = 0

                        BUY_COIN = Coin[Action] if Action !=len(Coin) else 'CNY'
                        SELL_COIN = Coin[self.ValueAccount] if self.ValueAccount !=len(Coin) else 'CNY'

                        print('Time',number, 'Buy %s' % BUY_COIN,'Price',Price)
                        print('Sell %s' % SELL_COIN, 'Price', SellPrice, 'Current_Profit', Cny - Initial_Asset)

                        names['QTY%s' % Action] = (Cny / Price) * 0.998
                        now = datetime.datetime.now()
                        now = now.strftime('%Y-%m-%d %H:%M:%S')
                        f = open(Trade_Path_Testback, 'r+')
                        f.read()
                        f.write('\nTime %s' % number)
                        f.write('\nSell %s , Price %s, Current_Profit %s' % (
                            SELL_COIN, SellPrice, Cny - Initial_Asset))
                        f.write('\nBuy %s , Price %s' % (BUY_COIN, Price))
                        f.close()

                self.Price_Begun = Price
                self.ValueAccount = Action

        number = -1
        CoinPrice = 0
        for y in range(len(Coin)+1):
            CoinPrice += self.PriceArray[number,y] * names['QTY%s' % y]
            if names['QTY%s' % y] > 0:
                CoinName = Coin[y] if y !=len(Coin) else 'CNY'
                print('%s QTY' % CoinName, names['QTY%s' % y], ' Last_Price %s' % self.PriceArray[number,y])
                break
        profit = CoinPrice - Initial_Asset

        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print(now, 'Profit:%f' % profit, 'Total Asset:%f' % (profit + Initial_Asset))

        f = open(Trade_Path_Testback, 'r+')
        f.read()
        f.write('%s,Profit:%f ,Total Asset:%f' % (now,profit,profit + Initial_Asset))
        f.close()

if __name__ == '__main__':

    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')

    if Read_Trade_Log :

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
        QTY = float(Total_Asset / float(str(ValueAccount_Txt[-1]).split(' ')[-1])) * 0.998
        print('Successfully loaded:Trade_Log')

    else:

        ValueAccount = 'CNY'
        from Class_Trade import Trade_Api
        Trade_api = Trade_Api()
        Initial_Asset = Trade_api.GetAsset()
        QTY = float(Initial_Asset / Get_Data._USDT_CNY)
        Price_Begun = Get_Data._USDT_CNY
        print("Initial Model")

    try:
        action_last = Coin.tolist().index(ValueAccount)
    except:
        action_last = len(Coin)

    names = locals()
    for x in range(len(Coin)+1):
        names['QTY%s' % x] = 0

    names['QTY%s' % action_last] = QTY

    TestBack = TestBack( action_last=action_last,Price_Begun=Price_Begun)
    TestBack.Run_Testback()


