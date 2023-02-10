import socket
import time

HOST = "192.168.2.200"
PORT = 30002


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

time.sleep(2)

while True:

	s.send(("movej([1.574,-1.487,1.606,-1.695,-1.578,0.789],a=1.396,v=1.047)" +"\n").encode('utf8'))