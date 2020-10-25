import sqlite3
import json
import time

DATABASE = 'Sqlite3/database.db'

#接受json输入
def db_add_item(post_data):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT MAX(ID) FROM ACCRWA")
    id = cur.fetchone()[0] + 1
    name = post_data.get("username", 'None')
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime());
    lon = post_data.get("lon", 0)
    lat = post_data.get("lat", 0)
    img_path = post_data.get("img_path", 0)
    lab_path = post_data.get("lab_path", 0)
    cur.execute("INSERT INTO ACCRWA (ID,NAME,TIME,LON,LAT,IMG_PATH,LAB_PATH) \
      VALUES (?, ?, ?, ?, ?, ?, ?)",(id,name,now_time,lon,lat,img_path,lab_path))
    cur.close()
    conn.commit()
    conn.close()
    print('Execute success')

def db_delete_item(id):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("delete from ACCRWA where ID=?",(id,))
    cur.close()
    conn.commit()
    conn.close()








