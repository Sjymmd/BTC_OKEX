#!/use/bin/env python
#coding=utf-8
import itchat
from itchat.content import *


@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    print(msg)
    ToUserName = msg['ToUserName']
    return ToUserName

class Wechat():

    def __init__(self,Dataframe,ToUserName):
        itchat.auto_login(hotReload=True)
        self.ToUserName = ToUserName
        itchat.send(Dataframe, ToUserName)
        # itchat.run()

    def msg(self,msg):
        itchat.send(msg,self.ToUserName)

if __name__ == '__main__':
    Wechat = Wechat('Test','@@98e2290e631e5dceb8d91aab05775454e78f94640ba3ab2c7e7de23c2840f6b6')
    Wechat.msg('k')
    # itchat.send('abc', ToUserName)