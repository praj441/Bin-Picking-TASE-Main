from multiprocessing.connection import Client
import time

address = ('localhost', 6003)
conn = Client(address, authkey=b'secret password')
while True:
	time.sleep(10)
conn.send('1')