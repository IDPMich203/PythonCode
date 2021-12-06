import socket
import time
import random


s = socket.socket()

s.bind(('0.0.0.0', 8090))
s.listen(0)

# This makes code run faster but accept() and recv() throw errors if no data available hence try/excepts
s.setblocking(False)

# Honestly not sure why this loop is there
while True:

    # Get connected
    while True:
        try:
            client, addr = s.accept()
            break
        except:
            pass

    # Get content, have some useless prints and send either on or off to the Arduino.
    while True:
        time.sleep(1)
        print("starting recv")
        try:
            content = client.recv(32)
        except:
            pass
        print("Ended recv")
        print(content)

        if(random.random() > 0.5):
            client.send(b"on\n")
        else:
            client.send(b"off\n")

    # Don't know what to do with this but this is how you close the connection
    print("Closing connection")
    client.close()
    break
