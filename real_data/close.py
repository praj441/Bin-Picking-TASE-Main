from multiprocessing.connection import Client

address = ('localhost', 6003)
conn = Client(address, authkey=b'secret password')
conn.send('close')