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

    def Get_ClassifierData(self):

        from Volume_Early_Warning import Okex_Api

        Okex_Api = Okex_Api()
        Okex_Api._Lenth = 24 * 100
        Coin = Okex_Api.GetCoin()
        names = locals()
        StartTime = time.time()
        USDT_CNY = Okex_Api._USDT_CNY

        print('Start Loading Data...')

        StartTime = time.time()

        Coin = self.Coin

        DataLen = []

        scaler = preprocessing.StandardScaler()
        for x in Coin:

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

        Price_Begun = USDT_CNY

        Reward = 0
        Q_Value = 0
        Action_last = 0
        D_price = 0
        ValueAccount = 'CNY'

        ClassifierData = pd.DataFrame()

        for i in range(2,len(Data)):

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
            d_price = max(DataNow[:i+1]) - DataNow[i]
            # print(i,len(Data) - 1, action, reward, agent.Q_Value)

            Price = [USDT_CNY] if CoinName=='CNY' else names['TestPrice%s' % CoinName][i]
            Price = Price[0]
            SellPrice = [USDT_CNY] if ValueAccount =='CNY' else names['TestPrice%s' % ValueAccount][i]
            SellPrice = SellPrice[0]

            if CoinName != ValueAccount :

                Inc= float(SellPrice / Price_Begun - 1)

                Target = Inc if  Price_Begun < SellPrice else 0

                Price_Begun = Price

                ValueAccount = CoinName

                insert = np.array([Reward, Q_Value,Action_last,D_price])
                insert = scaler.fit_transform(insert.reshape((-1,1))).reshape(insert.shape[0],)

                D_price = d_price
                Reward = reward
                Q_Value = agent.Q_Value
                Action_last = action

                ClassifierDa = (state.tolist()+insert.tolist()+[Target])
                # print(ClassifierDa)
                ClassifierDa = np.array([ClassifierDa])
                # ClassifierDa = ClassifierDa.reshape(len(state)+len(insert),)
                if i ==2:
                    ClassifierData = ClassifierDa
                else:
                    ClassifierData = np.row_stack((ClassifierData,ClassifierDa))


        np.savetxt('./ClassifierData_Classifier.csv', ClassifierData, delimiter=',')

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

    Classifier = Classifier()

    while True:

        # while True:
        #         timenow = datetime.datetime.now()
        #         minutes = timenow.minute
        #         if minutes > 25 and minutes < 40:
        #             break
        #         time.sleep(10*60)

        # Data = np.loadtxt(open("./ClassifierData_Classifier.csv", "rb"), delimiter=",", skiprows=0)
        Data = Classifier.Get_ClassifierData()
        Data_Target = Data[:, -1].reshape(-1, 1)
        from sklearn.cluster import KMeans

        k = 5
        estimator = KMeans(n_clusters=k)
        estimator.fit(Data_Target)
        label = estimator.fit_predict(Data_Target)

        expenses = np.sum(estimator.cluster_centers_, axis=1)
        expenses = np.array(expenses)
        expense = expenses.tolist()
        expense_index = expenses.tolist()
        expense_index.sort()
        Best = []
        for x in expense_index:
            Best.append(expense.index(x))

        Target = np.zeros((len(label), k))

        for x in range(len(label)):
            Target[x, Best.index(label[x])] = 1


        Feature_Train,Feature_Test,Target_Train\
            ,Target_Test = train_test_split(Data[:,:-1],Target
                                            ,test_size = 0.2,  random_state = 0)

        while True:
            try:
                model = load_model('./Keras_Model/my_model_classifier.h5')
                print('Successfully loaded : my_model_classifier')
                break
            except:
                model = Sequential()
                model.add(Dense((Data.shape[1]-1)*2,input_dim=Data.shape[1]-1, init='uniform', activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(Data.shape[1]-1, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(Data.shape[1]-1, activation='relu'))
                model.add(Dense(k, init='uniform', activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                break

        model.fit(Feature_Train, Target_Train,batch_size=10, nb_epoch=10000, verbose=0,
              validation_data=(Feature_Test, Target_Test))
        model.save('./Keras_Model/my_model_classifier.h5')

        from keras.models import load_model
        model = load_model('./Keras_Model/my_model_classifier.h5')

        score = model.evaluate(Feature_Test, Target_Test,)
        # print(model.predict_classes(Feature_Test))
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        tf.reset_default_graph()
        import gc
        gc.collect()


