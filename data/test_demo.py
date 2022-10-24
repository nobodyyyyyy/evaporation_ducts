import sqlite3

conn = sqlite3.connect('hxywq.db3')
cursor=conn.cursor()
cursor.execute("SELECT Time,Temperature1,Humidity1,TemperatureSea,Pressure,WinspeedR1 from 实时数据")
result=cursor.fetchall()
print(type(result[-1][2]))
#for row in c:
#	print(row[0])
