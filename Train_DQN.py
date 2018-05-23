# coding: utf-8
from sklearn import preprocessing
from Model_DQN import *
import warnings
import pickle
warnings.filterwarnings("ignore")

Coin = np.loadtxt("./Log/Coin_Select.txt",dtype=np.str)
EPISODE = 10000*100
TEST = 1

def Main():

    # Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)
    with open('./Data/Data.pickle', 'rb') as myfile:
        Data = pickle.load(myfile)
    Data = scaler.fit_transform(Data)
    # PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)
    with open('./Data/PriceArray.pickle', 'rb') as myfile:
        PriceArray = pickle.load(myfile)

    lenth = int(Data.shape[0] * 5 / 6)
    STEP = lenth - 1
    # my_train = Data[:lenth]
    my_train = Data
    my_test = Data[lenth:]

    env = TWStock(my_train)
    agent = DQN()

    print('Start Training...')
    train_output = ""
    rate_string = ""
    Total_Train_reward = 0
    Total_rate = 0
    loss = 0
    for episode in range(EPISODE):

        # initialize task
        state = env.reset()
        # Train
        out = "train\n"
        train_reward = 0

        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for trai
            env.stock_rewards = PriceArray[:,action]
            next_state, reward, done, _ = env.step(action)
            out += str(reward) + " "
            train_reward += reward
            # Define reward for agent
            # reward_agent = -1 if done else 0.1
            # print(step,STEP,action,reward,agent.Q_Value)
            agent.perceive(state, action, reward, next_state, done)
            agent.store_transition(state, action, np.float64(reward), next_state)
            state = next_state
            if done:
                lossnew = agent.learn()
                loss +=lossnew
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
        if episode % 100 == 0:

            out = "test\n"
            env1 = TWStock(my_test)
            total_reward = 0
            for i in range(TEST):
                state = env1.reset()

                for j in range(STEP):
                    env1.render()
                    action = agent.action(state)  # direct action for test
                    env1.stock_rewards = PriceArray[:,action]
                    state, reward, done, _ = env1.step(action)
                    out += str(action) + " " + str(reward) + ","
                    total_reward += reward

                    if done:
                        break
            # fo.write(out + "\n")
            count = 1 if episode ==0 else 10
            ave_reward = total_reward / TEST
            ave_train_rewards =Total_Train_reward/count
            ave_rate =Total_rate/ count
            ave_loss = loss/count
            # print(train_output)
            # print('Train_Rewards',ave_train_rewards, 'Training Rate:', ave_rate,'Loss',ave_loss)
            train_output = ""
            Total_rate = 0
            Total_Train_reward = 0
            loss = 0
            # print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            rate_string = ""
        # tf.summary.merge_all()
        # tf.summary.FileWriter('./Log')

            # Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)
            with open('./Data/Data.pickle', 'rb') as myfile:
                Data = pickle.load(myfile)
            Data = scaler.fit_transform(Data)
            # PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)
            with open('./Data/PriceArray.pickle', 'rb') as myfile:
                PriceArray = pickle.load(myfile)
            lenth = int(Data.shape[0] * 5 / 6)
            STEP = lenth - 1
            # my_train = Data[:lenth]
            my_train = Data
            my_test = Data[lenth:]
            env.stock_data = my_train
            agent.saver.save(agent.session, './DQN_Model/' + 'network' + '-dqn', global_step=agent.time_step)

        if episode %1000 ==0 and episode!= 0:
            agent.INITIAL_EPSILON = 0.5

if __name__ == '__main__':


    while True:

        scaler = preprocessing.StandardScaler()

        import time
        StartTime = time.time()
        tf.reset_default_graph()
        Main()
        EndTime = time.time()
        print('Training Using_Time: %d min' % int((EndTime - StartTime) / 60))



