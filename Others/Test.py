# import numpy as np
# Coin = np.loadtxt("Coin_Select.txt", dtype=np.str)
# print(Coin.tolist().index('itc_usdt'))


if __name__ == '__main__':

    Trade_Path = 'Trade_Log.txt'
    f = open(Trade_Path, 'r+')
    ValueAccount_Txt = f.readlines()
    # f.read()
    # f.write('CreateTime %s' % now)
    f.close()

    Price_Begun = float(str(ValueAccount_Txt[-1]).split(' ')[-1])