from Class_OKEX_API import *

Okex_Api = Okex_Api()
Okex_Api._Lenth = 24*10
class Get_Data():

    def __init__(self):

        self._USDT_CNY = Okex_Api._USDT_CNY
        self.Coin = np.loadtxt("./logs/Coin_Select.txt", dtype=np.str).tolist()

    def GetData(self,save = False):

        print('Start Loading Data...')

        StartTime = time.time()
        Coin = self.Coin
        DataLen =[]
        for x in Coin:

            while True:
                try:
                    Data_Initial = Okex_Api.GetDataCoin(x)
                except:
                    # print('Get_Dataframe Error')
                    time.sleep(10)
                    continue
                if Data_Initial is not None:
                    break

            # print(Data_Initial)

            # Data_Pre = scaler.fit_transform(Data_Initial.iloc[:, 1:].as_matrix())
            Data_Pre = Data_Initial.iloc[:, 1:].as_matrix()
            PriceArray_Pre= Data_Initial.iloc[:, 1].reshape(-1, 1)

            if Coin.index(x) == 0:
                Data = Data_Pre
                PriceArray = PriceArray_Pre
            else:
                Data = np.column_stack((Data, Data_Pre))
                PriceArray = np.column_stack((PriceArray, PriceArray_Pre))

            DataLen.append(Data_Pre.shape[0])

        self.lenData = min(DataLen)

        Cny_Price = pd.DataFrame(columns=['A'])

        for x in range(self.lenData):
            Cny_Price = Cny_Price.append({'A': self._USDT_CNY}, ignore_index=True)
        Cny_Price = Cny_Price['A'].reshape(-1, 1)
        PriceArray = np.column_stack((PriceArray,Cny_Price))

        if save is True:
            np.savetxt('./Data/PriceArray.csv', PriceArray, delimiter=',')
            np.savetxt('./Data/Data.csv',Data,  delimiter=',')
            print('Saved Data Successfully')

        print('Loading Data Using_Time: %d min' % int((time.time() - StartTime) / 60))
        return Data,PriceArray

    def GetData_Now(self):

        Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)
        PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)
        Data_Insert, PriceArray_Insert = Get_Data.GetData(self)
        Data =np.vstack((Data,Data_Insert[-1,:]))
        PriceArray = np.vstack((PriceArray,PriceArray_Insert[-1,:]))
        np.savetxt('./Data/PriceArray.csv', PriceArray, delimiter=',')
        np.savetxt('./Data/Data.csv', Data, delimiter=',')
        print('Update Data Successfully')

if __name__ == '__main__':

    Get_Data = Get_Data()
    Get_Data.GetData(save=True)