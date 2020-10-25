import sqlite3
import json
import time
from config import *
import os

#接受json输入
def db_add_item(post_data):
	conn = sqlite3.connect(DB_PATH)
	print("Connect success")
	cur = conn.cursor()
	cur.execute("SELECT MAX(ID) FROM ACCRWA")
	id = cur.fetchone()[0] + 1
	name = post_data.get("username", 'None')
	now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime());
	lon = post_data.get("lon", 0)
	lat = post_data.get("lat", 0)
	img_path = post_data.get("img_path", '')
	lab_path = post_data.get("lab_path", '')
	score = post_data.get("score", 0)
	cur.execute("INSERT INTO ACCRWA (ID,NAME,TIME,LON,LAT,IMG_PATH,LAB_PATH,SCORE) \
	  VALUES (?, ?, ?, ?, ?, ?, ?, ?)",(id,name,now_time,lon,lat,img_path,lab_path,score))
	cur.close()
	conn.commit()
	conn.close()
	print('Execute success')
	return id

def db_delete_item(id):
    conn = sqlite3.connect(DB_PATH)
    print("Connect success")
    cur = conn.cursor()
    cur.execute("delete from ACCRWA where ID=?",(id,))
    cur.close()
    conn.commit()
    conn.close()
    print('Execute success')

def db_add_score(post_data):
	conn = sqlite3.connect(DB_PATH)
	print("Connect success")
	cur = conn.cursor()
	score = post_data.get("score", 0)
	id = post_data.get("id")
	cur.execute("UPDATE ACCRWA set SCORE = ? where ID=?", (score, id), )
	cur.close()
	conn.commit()
	conn.close()
	print('Execute success')

'''def db_add_score(id,score):
	conn = sqlite3.connect(DB_PATH)
	print("Connect success")
	cur = conn.cursor()
	cur.execute("UPDATE ACCRWA set SCORE = ? where ID=?", (score,id),)
	cur.close()
	conn.commit()
	conn.close()
	print('Execute success')'''











