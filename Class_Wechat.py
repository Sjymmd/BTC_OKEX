#!/use/bin/env python
#coding=utf-8
import itchat
# from itchat.content import *

class Wechat():

    def __init__(self):
        itchat.auto_login(hotReload=True)
        # itchat.run()

    def Get_Chatrooms(self,Name):
        group = itchat.get_chatrooms(update=True)
        for g in group:
            if g['NickName'] == Name:
                self.from_group = g['UserName']

    def Get_Friends(self,Name):
        friend = itchat.get_friends(update=True)
        for f in friend:
            if f['NickName'] == Name:
                self.friend = f['UserName']



    def msg(self,wechatmsg,ToUser):
        itchat.send(wechatmsg, ToUser)

if __name__ == '__main__':
    Wechat = Wechat()
    # Wechat.Get_Chatrooms('Python')
    # print(Wechat.from_group)
    group=Wechat.Get_Chatrooms('PythonGroup')
    friend=Wechat.Get_Friends('belief.')
    print(Wechat.from_group,Wechat.friend)