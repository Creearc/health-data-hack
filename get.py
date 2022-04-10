# -*- coding: cp1251 -*-
import threading
from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

def main_thread():
  ftp = FTP()
  HOST = '192.168.68.202'
  PORT = 21

  ftp.connect(HOST, PORT)

  print(ftp.login(user='alexandr', passwd='9'))

  ftp.cwd('yolact-repo/weights')

  for i in ['hdh_6_14000.pth',
            'hdh_113_227000.pth',
            'hdh_100_201000.pth',
            'hdh_74_149000.pth'
            ]:
    try:
      with open(i, 'wb') as f:
        ftp.retrbinary('RETR ' + i, f.write)
    except:
      print(i)


main_thread()

