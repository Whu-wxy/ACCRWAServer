import sqlite3
import json
import time

DATABASE = 'Sqlite3/database.db'

conn = sqlite3.connect(DATABASE)
c = conn.cursor()
c.execute('''CREATE TABLE ACCRWA
       (ID INT PRIMARY KEY     NOT NULL,
       NAME           TEXT    ,
       TIME            CHAR(50),
       LON        DECIMAL(10,7),
       LAT        DECIMAL(10,7),
       IMG_PATH        CHAR(50),
       LAB_PATH        CHAR(50)
       );''')
print("Execute success")
conn.commit()
conn.close()