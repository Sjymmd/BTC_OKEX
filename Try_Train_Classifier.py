from sklearn import preprocessing
from Model_DQN_Increase import *
from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import warnings

warnings.filterwarnings("ignore")

class Classifier():

    def __init__(self):
        # self.Trade_Path = './logs/Trade_TestBack.txt'
        self.Coin = np.loadtxt("./logs/Coin_Select.txt", dtype=np.str)
        self.Total_Asset = 400

    def Get_ClassifierData(self):

        from Volume_Early_Warning import Okex_Api

        Okex_Api = Okex_Api()
        Coin = Okex_Api.GetCoin()
        names = locals()
        StartTime = time.time()

        USDT_CNY = okcoinfuture.exchange_rate()['rate']

        print('Start Loading Data...')

        StartTime = time.time()

        Coin = self.Coin

        DataLen = []

        for x in Coin:
            scaler = preprocessing.StandardScaler()
            while True:
                try:
                    TestData = Okex_Api.GetDataCoin(x)
                except:
                    print('Get_Dataframe Error')
                    time.sleep(5)
                    continue
                if TestData is not None:
                    break
                print('Get %s error'%x)
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

        print('MinLenth',lenData)
        Tem = names['TestData%s' % Coin[0]]
        Tem_Price = names['TestPrice%s' % Coin[0]]
        names['TestPrice%s' % Coin[0]] = Tem_Price[int(len(Tem) - lenData):]
        names['TestData%s' % Coin[0]] = Tem[int(len(Tem)-lenData):]
        Data = names['TestData%s' % Coin[0]]
        for x in Coin[1:]:
            Tem = names['TestData%s' % x]
            Tem_Price = names['TestPrice%s' % x]
            names['TestPrice%s' % x] = Tem_Price[int(len(Tem) - lenData):]
            names['TestData%s' % x] = Tem[int(len(Tem)-lenData):]
            Data = np.column_stack((Data, names['TestData%s' % x]))

        EndTime = time.time()
        print('Loading Data Using_Time: %d min' % int((EndTime - StartTime) / 60))

        agent = DQN()
        Total_Asset  = self.Total_Asset

        for x in Coin:
            names['Amount%s' % x] = 0
        names['AmountCNY']=Total_Asset /USDT_CNY
        Price_Begun = USDT_CNY

        Profit = 0
        Reward = 0
        Q_Value = 0
        Action_last = 0

        ClassifierData = pd.DataFrame(columns=(
            "Reward", "Q_Value", "Target"))

        for i in range(2,len(Data)):
            for x in Coin:
                if names['Amount%s' % x] >0:
                    ValueAccount = x
                elif names['AmountCNY']>0:
                    ValueAccount = 'CNY'

            state = Data[i]
            action = agent.action(state)  # direct action for test
            CoinName = Coin[action] if action < len(Coin) else 'CNY'
            DataNow = names['TestPrice%s' % CoinName]

            gamma = 0.95
            fex = 1 / (0.998 * 0.998) - 1
            f_reward = 0

            for x in range(0, 2):

                f_reward += gamma * (((DataNow[i - x] - DataNow[i - x - 1]) / DataNow[i - x -1]) - fex)
                gamma = gamma ** 2

            # action_reward = (NowData*gamma + f_reward)*count
            action_reward = float(f_reward)
            reward = action_reward
            # print(i,len(Data) - 1, action, reward, agent.Q_Value)

            Price = [USDT_CNY] if CoinName=='CNY' else names['TestPrice%s' % CoinName][i]
            Price = Price[0]
            SellPrice = [USDT_CNY] if ValueAccount =='CNY' else names['TestPrice%s' % ValueAccount][i]
            SellPrice = SellPrice[0]

            if CoinName != ValueAccount :
                Cny = names['Amount%s' % ValueAccount] * SellPrice*0.998
                names['Amount%s' % ValueAccount] = 0

                Target = 1 if  Profit <  Cny - Total_Asset else 0

                Price_Begun = Price

                names['Amount%s' % CoinName] = (Cny / Price)*0.998
                ValueAccount = CoinName
                Profit = Cny - Total_Asset

                insert = np.array([Reward, Q_Value,Action_last,Target])

                Reward = reward
                Q_Value = agent.Q_Value
                Action_last = action

                ClassifierDa = (state.tolist()+insert.tolist())
                ClassifierDa = np.array([ClassifierDa])
                # ClassifierDa = ClassifierDa.reshape(len(state)+len(insert),)
                if i ==2:
                    ClassifierData = ClassifierDa
                else:
                    ClassifierData = np.row_stack((ClassifierData,ClassifierDa))
        return ClassifierData


if __name__ == '__main__':

    import time
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import warnings
    from keras.models import load_model

    warnings.filterwarnings("ignore")
    while True:

        while True:
                timenow = datetime.datetime.now()
                minutes = timenow.minute
                if minutes > 25 and minutes < 40:
                    break
                time.sleep(10*60)

        Classifier = Classifier()
        Data = Classifier.Get_ClassifierData()

        Feature_Train,Feature_Test,Target_Train\
            ,Target_Test = train_test_split(Data[:,:-1],Data[:,-1]
                                            ,test_size = 0.2,  random_state = 0)

        while True:
            try:
                model = load_model('./Keras_Model/my_model.h5')
                print('Successfully loaded : my_model')
                break
            except:
                model = Sequential()
                model.add(Dense(Data.shape[1]-1*2,input_dim=Data.shape[1]-1, init='uniform', activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(Data.shape[1]-1, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(Data.shape[1]-1, activation='relu'))
                model.add(Dense(1, init='uniform', activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                break

        model.fit(Feature_Train, Target_Train,batch_size=10, nb_epoch=10000, verbose=0,
              validation_data=(Feature_Test, Target_Test))
        model.save('./Keras_Model/my_model.h5')

        from keras.models import load_model
        model = load_model('./Keras_Model/my_model.h5')

        score = model.evaluate(Feature_Test, Target_Test,)
        # print(model.predict_classes(Feature_Test))
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

