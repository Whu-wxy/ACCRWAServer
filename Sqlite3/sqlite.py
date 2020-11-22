import sqlite3
import json
import time
import  sys
sys.path.append('../')
from config import *
import os

#接受dict输入
def db_add_item(post_data):
	try:
		conn = sqlite3.connect(DB_PATH)
		print("Connect success")
		cur = conn.cursor()
		try:
			cur.execute("SELECT MAX(ID) FROM ACCRWA")
			res = cur.fetchone()
			id = res[0] + 1
		except:
			id = 1
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
		print('db_add_item success.')
		return id
	except:
		print('db_add_item fail.')
		return -1

def db_delete_item(id):
	try:
		conn = sqlite3.connect(DB_PATH)
		print("Connect success")
		cur = conn.cursor()
		cur.execute("delete from ACCRWA where ID=?",(id,))
		cur.close()
		conn.commit()
		conn.close()
		print('db_delete_item success.')
	except:
		print('db_delete_item fail.')

def db_add_score(post_data):
	try:
		conn = sqlite3.connect(DB_PATH)
		print("Connect success")
		cur = conn.cursor()
		score = post_data.get("score", 0)
		id = post_data.get("id")
		cur.execute("UPDATE ACCRWA set SCORE = ? where ID=?", (score, id), )
		cur.close()
		conn.commit()
		conn.close()
		print('db_add_score success.')
	except:
		print('db_add_score fail.')


def db_query_by_id(id):
	try:
		data = {}
		conn = sqlite3.connect(DB_PATH)
		cur = conn.cursor()
		cur.execute("SELECT * FROM ACCRWA WHERE ID = ?",(id,))
		results = cur.fetchall()
		for row in results:
			data['ID'] = row[0]
			data['NAME'] = row[1]
			data['TIME'] = row[2]
			data['LON'] = row[3]
			data['LAT'] = row[4]
			data['IMG_PATH'] = row[5]
			data['LAB_PATH'] = row[6]
			data['score'] = row[7]
		cur.close()
		conn.commit()
		conn.close()
		print("db_query_by_id success.")
		return data
	except:
		print('db_query_by_id fail.')
		return {}


#按姓名查询，用于小程序查询同一用户所有信息
def db_query_by_name(name):
	try:
		data = {}
		conn = sqlite3.connect(DB_PATH)
		cur = conn.cursor()
		cur.execute("SELECT * FROM ACCRWA WHERE NAME = ?",(name,))
		results = cur.fetchall()
		for row in results:
			data['ID'] = row[0]
			data['NAME'] = row[1]
			data['TIME'] = row[2]
			data['LON'] = row[3]
			data['LAT'] = row[4]
			data['IMG_PATH'] = row[5]
			data['LAB_PATH'] = row[6]
			data['score'] = row[7]
		json_data = json.dumps(data)
		cur.close()
		conn.commit()
		conn.close()
		print("db_query_by_name success.")
		return data
	except:
		print('db_query_by_name fail.')
		return {}

def db_delete_by_time(time):
	try:
		conn = sqlite3.connect(DB_PATH)
		cur = conn.cursor()
		cur.execute("delete from ACCRWA where TIME <?",(time,))
		cur.close()
		conn.commit()
		conn.close()
		print('db_delete_by_time success.')
	except:
		print('db_delete_by_time fail.')


###########################################################################

#添加数据，接受json输入，用于文本数据库添加数据
def word_add_item(post_data):
	try:
		conn = sqlite3.connect(WORD_DB_PATH)
		cur = conn.cursor()
		word = post_data.get("word", 'None')
		oldword = post_data.get("oldword", 'Error')
		strokes = post_data.get("strokes", 0)
		pinyin = post_data.get("pinyin", 'Error')
		radicals = post_data.get("radicals", 'Error')
		explanation = post_data.get("explanation", 'Error')
		cur.execute("INSERT OR IGNORE INTO WORDTABLE (WORD,OLDWORD,STROKES,PINYIN,RADICALS,EXPLANATION) \
		  VALUES (?, ?, ?, ?, ?, ?)",(word,oldword,strokes,pinyin,radicals,explanation))
		cur.close()
		conn.commit()
		conn.close()
		print('word_add_item success.')
	except:
		print('word_add_item fail.')



##按字查询，用于查找文本数据库中相应的汉字信息
def word_name_query_item(word):
	try:
		data = {}
		conn = sqlite3.connect(WORD_DB_PATH)
		cur = conn.cursor()
		cur.execute("SELECT * FROM WORDTABLE WHERE WORD = ?",(word,))
		results = cur.fetchall()
		for row in results:
			data['WORD'] = row[0]
			data['OLDWORD'] = row[1]
			data['STROKES'] = row[2]
			data['PINYIN'] = row[3]
			data['RADICALS'] = row[4]
			data['EXPLANATION'] = row[5]
		cur.close()
		conn.commit()
		conn.close()
		print('word_name_query_item success.')
		return data
	except:
		print('word_name_query_item fail.')

if __name__ == "__main__":
	db_delete_by_time('2020-11-05 22:34:27')






