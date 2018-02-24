# coding: utf-8
from DQN_Model import *

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24*1000
Okex_Api._EndLenth = 0


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
        print('%sError'%Coin)

# Hyper Parameters
EPISODE = 200  # Episode limitation
# 300 # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode

def Main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)

    env = TWStock(my_train)
    agent = DQN(env)

    print('Start')
    train_output = ""
    rate_string = ""
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
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
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
            print(train_output)
            train_output = ""
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward, 'training Rate past10:', rate_string)
            rate_string = ""
            # if ave_reward >= 1000:
            #     print('End')
            #     break
    print('Train_Output',train_output)
    print('episode: ', episode, 'Evaluation Average Reward:', ave_reward, 'training Rate past10:', rate_string)





if __name__ == '__main__':

    # Coin = ['snt_usdt']
    for x in Coin[:int(Okex_Api._CoinLenth)]:
        try:
            DataFrame = Get_Dataframe(x)
            # print(DataFrame)
            if len(DataFrame) < 1000:
                print('%s less than 1000 lines' % x)
                continue
            Data = DataFrame.iloc[:, 1:]
            lenth = int(len(Data) * 5 / 6)
            STEP = lenth - 1
            Train = Data.iloc[:lenth, ]
            my_train = scaler.fit_transform(Train)
            Test = Data.iloc[lenth:, ]
            my_test = scaler.fit_transform(Test)
            # y = open("./TXT/%s.txt"%x, "w")
            # y.close()
            # fo = open("./TXT/%s.txt"%x, "a")
            tf.reset_default_graph()
            print(x)
            Main()

        except:
            continue

    from DQN_Select import *
    DQN_Select = DQN_Select()
    DQN_Select.DQN_Select()