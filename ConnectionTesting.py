# Example script to test the Connection class with inputs

from Connection import Connection
import queue

from_arduino = queue.Queue()
to_arduino = queue.Queue()

conn = Connection(from_arduino, to_arduino)

conn.daemon = True
conn.start()

while True:
    while not from_arduino.empty():
        print(from_arduino.get_nowait())

    to_arduino.put(input("Enter message: "))
