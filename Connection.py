import threading
import socket
import os
import time
import random


class Connection(threading.Thread):
    """Connection class that provides an interface to connect to an arduino over sockets. """

    def __init__(self, from_arduino, to_arduino, port=8090, host="0.0.0.0"):
        super(Connection, self).__init__()
        self.from_arduino = from_arduino
        self.to_arduino = to_arduino
        self.port = port
        self.host = host
        # Exceptions aren't thrown from threads to an interface has to be made to get the errors
        self.exception = None

    def run(self):
        # Try to open the socket server
        print("Opening socket server")
        s = socket.socket()
        s.bind(('0.0.0.0', 8090))
        s.listen(0)
        # Non blocking so everything throws an error which is why there are so many try/except blocks
        s.setblocking(False)

        client, addr = None, None

        lastdata = ""
        while True:
            # Try accepting a connection
            try:
                client, addr = s.accept()
                print("Client connected: ", addr)
                break
            except:
                pass

        # Forever
        i = 0
        while True:
            i += 1
            time.sleep(0.1)
            try:
                # Recieve data and decode it
                content = client.recv(128)
                print(content.decode('ascii'))
                lastdata += content.decode("ascii")

                # Split into commands if multiple were recieved
                if("\n" in lastdata):
                    data = lastdata.split("\n")
                    lastdata = "\n".join(data[1:])
                    # Ignore messages
                    if(data.startswith("m")):
                        print(data)
                    # Put to queue
                    else:
                        self.from_arduino.put(data[0])
                        print("Put: ", data[0])
            except:
                pass

            if(self.to_arduino.qsize()):
                print("Sending data")
                data: str = self.to_arduino.get()
                if not data.endswith("\n"):
                    data += "\n"
                    client.send(bytes(data, 'ascii'))
                else:
                    client.send(bytes(data, "ascii"))

        print("Closing connection")
        client.close()
        return

    def get_exception(self):
        return self.exception
