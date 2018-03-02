# coding: utf-8
from __future__ import division

from sklearn import preprocessing
from Model_DQN_Increase import *
from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import warnings

warnings.filterwarnings("ignore")
Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24*1000
Okex_Api._EndLenth = 0
# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)
names = locals()
StartTime = time.time()

def Get_Dataframe(Coin):
    try:
        DataFrame = pd.DataFrame(columns=("Coin", "Cny","High","Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
        data = pd.DataFrame(okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChosen], Okex_Api._Lenth, Okex_Api._EndLenth, Coin)).iloc[:Okex_Api._Lenth-1, ]
        data[5] = data.iloc[:, 5].apply(pd.to_numeric)
        data = data[data[5] >= 1000]
        data = data.reset_index(drop=True)
        Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
        # Increase = str('%.2f' % (Increase) + '%')
        price = float(data.iloc[0, 4])
        Hi_price = round(float((data.iloc[0, 2]))* Okex_Api._USDT_CNY,2)
        Lo_price = round(float((data.iloc[0, 3]))* Okex_Api._USDT_CNY,2)
        Cny = round(price * Okex_Api._USDT_CNY, 2)
        Volume = float(data.iloc[0, 5])
        Volume_Mean = round(Volume/1000,2)
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
                # Increase = str('%.2f' % (Increase) + '%')
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
        return DataFrame
        # print(DataFrame)
    except:
        time.sleep(5)
        print('%s error'%Coin)

# ---------------------------------------------------------
def Coin_Select(Coin):
    for x in Coin:
        try:
            TestData = Get_Dataframe(x)

            if len(TestData) < 1000:
                Coin.remove(x)
                print('%s less than 1000 lines' % x)
                continue
        except:
            Coin.remove(x)
            continue


    np.savetxt("Coin_Select.txt", Coin,delimiter=" ", fmt="%s")
    Coin = np.loadtxt("Coin_Select.txt",dtype=np.str)
    print(len(Coin))
# ## main function

# Hyper Parameters
EPISODE = 100000  # Episode limitation
# 300 # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode

def TestBack():
    # Coin = pd.read_table('Coin_Select.txt', sep=',').iloc[:5, 0].values
    # Coin = ['btc_usdt','snt_usdt','eth_usdt']
    USDT_CNY = okcoinfuture.exchange_rate()['rate']
    print('Start Loading Data...')
    StartTime = time.time()
    Coin = np.loadtxt("Coin_Select.txt",dtype=np.str)
    DataLen = []
    for x in Coin:
        scaler = preprocessing.StandardScaler()
        while True:
            try:
                TestData = Get_Dataframe(x)
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
    print('MinLenth',lenData)
    Tem = names['TestData%s' % Coin[0]]
    names['TestData%s' % Coin[0]] = Tem[int(len(Tem)-lenData):]
    Data = names['TestData%s' % Coin[0]]
    for x in Coin[1:]:
        Tem = names['TestData%s' % x]
        names['TestData%s' % x] = Tem[int(len(Tem)-lenData):]
        Data = np.column_stack((Data, names['TestData%s' % x]))

    EndTime = time.time()
    print('Loading Data Using_Time: %d min' % int((EndTime - StartTime) / 60))

    env1 = TWStock(Data)
    state = env1.reset()
    agent = DQN(env1)
    Total_Asset  = 1000
    # Cny = Total_Asset
    for x in Coin:
        names['Amount%s' % x] = 0
    names['AmountCNY']=Total_Asset /round(USDT_CNY, 2)
    ValueAccount = 'CNY'
    for i in range(len(Data) - 1):
        for x in Coin:
            if names['Amount%s' % x] >0:
                ValueAccount = x
            elif names['AmountCNY']>0:
                ValueAccount = 'CNY'
        env1.render()
        action = agent.action(state)  # direct action for test
        state, reward, done, _ = env1.step(action)
        CoinName = Coin[action] if action<len(Coin) else 'CNY'
        Price = [USDT_CNY] if CoinName=='CNY' else names['TestPrice%s' % CoinName][i]
        Price = round(Price[0], 2)
        SellPrice = [USDT_CNY] if ValueAccount =='CNY' else names['TestPrice%s' % ValueAccount][i]
        SellPrice = round(SellPrice[0], 2)
        # Price = scaler_Price.inverse_transform(state[0].reshape(-1,1))

        if CoinName != ValueAccount and done is False :
            Cny = names['Amount%s' % ValueAccount] * SellPrice*0.998
            names['Amount%s' % ValueAccount] = 0
            print('Sell %s' % ValueAccount, 'Time', i, 'Price', SellPrice, 'Current_Profit', Cny - Total_Asset)
            print('Buy %s' % CoinName, 'Price', Price)
            names['Amount%s' % CoinName] = (Cny / Price)*0.998
            ValueAccount = CoinName
            Cny = 0
    CoinPrice = 0
    for x in Coin:
        CoinPrice += names['TestPrice%s' % x][-1] * names['Amount%s' % x]
        if names['Amount%s' % x] >0:
            print('%s Amount' % x, names['Amount%s' % x],' Last_Price %s'% names['TestPrice%s' % x][-1][0])
    if names['AmountCNY'] >0:
        CoinPrice += USDT_CNY * names['AmountCNY']
        print('AmountCNY', names['AmountCNY'], ' Last_Price %s' % USDT_CNY)
    profit = Cny + CoinPrice - Total_Asset
    print('Time',len(Data),'Profit:%d' % profit,'Total Asset:%d' %(profit+Total_Asset))

def Main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)

    env = TWStock(my_train)
    agent = DQN(env)

    print('Start Training...')
    train_output = ""
    rate_string = ""
    Total_Train_reward = 0
    Total_rate = 0
    for episode in range(EPISODE):

        # initialize task
        state = env.reset()

        # Train
        out = "train\n"
        train_reward = 0


        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for trai
            next_state, reward, done, _ = env.step(action)
            out += str(reward) + " "
            train_reward += reward
            # Define reward for agent
            # reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            agent.store_transition(state, action, np.float64(reward), next_state)
            state = next_state
            if done:
                agent.learn()
                break
        anal = out.split()
        p = 0.0
        n = 0.0

        for x in range(1, len(anal) - 1):
            if (float(anal[x]) > 0):
                p += float(anal[x])
            elif (float(anal[x]) < 0):
                n += float(anal[x])

        rate = round(p / (n * (-1) + p), 2)

        rate_string += str(rate) + " "
        # fo.write(out + "\n")
        train_output += str(train_reward) + " "
        Total_Train_reward +=train_reward
        Total_rate += rate

        # Test every 100 episodes
        if episode % 10 == 0:
            out = "test\n"
            env1 = TWStock(my_test)
            total_reward = 0
            for i in range(TEST):
                state = env1.reset()

                for j in range(STEP):
                    env1.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env1.step(action)
                    out += str(action) + " " + str(reward) + ","
                    total_reward += reward
                    if done:
                        break
            # fo.write(out + "\n")
            ave_reward = total_reward / TEST
            ave_train_rewards =Total_Train_reward/(episode+1)
            ave_rate =Total_rate/ (episode+1)
            # print(train_output)
            print('Train_Rewards',ave_train_rewards, 'Training Rate:', ave_rate)
            train_output = ""
            Total_rate = 0
            Total_Train_reward = 0
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            rate_string = ""

        tf.summary.merge_all()
        tf.summary.FileWriter('./logs')
            # if ave_reward >= 1000:
            #     print('End')
            #     break



if __name__ == '__main__':

    # Coin_Select(Coin)

    # print('Start Loading Data...')
    # Coin = np.loadtxt("Coin_Select.txt",dtype=np.str)
    # StartTime = time.time()
    # DataLen = []
    # for x in Coin:
    #     scaler = preprocessing.StandardScaler()
    #     while True:
    #         try:
    #             TestData = Get_Dataframe(x)
    #         except:
    #             print('Get_Dataframe Error')
    #             time.sleep(5)
    #             continue
    #         if TestData is not None:
    #             break
    #         print('Get %s error'%x)
    #     TestData = TestData.iloc[:, 1:]
    #     TestData_Initial = TestData.as_matrix()
    #     names['TestData%s' %x] = scaler.fit_transform(TestData_Initial)
    #     DataLen.append(names['TestData%s' %x].shape[0])
    # lenData = min(DataLen)
    # Tem = names['TestData%s' % Coin[0]]
    # names['TestData%s' % Coin[0]] = Tem[int(len(Tem)-lenData):]
    # Data = names['TestData%s' % Coin[0]]
    # for x in Coin[1:]:
    #     Tem = names['TestData%s' %x]
    #     names['TestData%s' % x] = Tem[int(len(Tem)-lenData):]
    #     Data = np.column_stack((Data, names['TestData%s' % x]))
    # lenth = int(Data.shape[0] * 5 / 6)
    # STEP = lenth - 1
    # my_train = Data[:lenth]
    # my_test = Data[lenth:]
    #
    #
    # EndTime = time.time()
    # print('Loading Data Using_Time: %d min' % int((EndTime - StartTime) / 60))
    #
    # StartTime = time.time()
    # tf.reset_default_graph()
    # Main()
    #
    # EndTime = time.time()
    # print('Training Using_Time: %d min' % int((EndTime - StartTime) / 60))

    TestBack()
