import threading
import serial
import os
import time


class SerialConnection(threading.Thread):
    """Serial connection class that provides an interface to read from and write to a
        serial port and write to a queue on a separate thread.
        Arguments: port{string} the port to connect to
                   baudrate{integer} the baudrate
                   q{queue.Queue object} the queue to write """

    def __init__(self, port, baudrate, recieve_queue, send_queue):
        super(SerialConnection, self).__init__()
        self.recieve_queue = recieve_queue
        self.send_queue = send_queue
        self.port = port
        self.baudrate = baudrate
        # Exceptions aren't thrown from threads to an interface has to be made to get the errors
        self.exception = None
        # Represent the last valid data point that was recieved
        self.lastdata = [0.0, 0.0]

    def run(self):
        # Try to open the serial port
        print("Opening Serial Port....")
        try:
            self.conn = serial.Serial(port=self.port, baudrate=self.baudrate)
            print("Reading from port %s..." % self.port)
        # Exit all threads if could not connect
        except Exception as e:
            print(
                "Error when opening port. Check Spellings and if there is a device connected")
            self.exception = e
            print(self.exception)
            os._exit(1)
        self.conn.flush()
        self.conn.reset_input_buffer()

        # Forever
        while True:
            raw = self.conn.readline()
            try:
                # Format the data to a list of floats
                data = raw.decode("ascii").strip()
                self.lastdata = data
            except Exception as e:
                print("There was an issue converting:", raw)
                self.exception = e
            # Write to queue
            self.recieve_queue.put(self.lastdata)

            if(self.send_queue.qsize()):
                data = self.send_queue.get()
                self.conn.write(data)
            # Monitor serial buffer
            if self.conn.in_waiting > 100:
                print("!!!Danger, Higher than average serial buffer. You will experience lag %i" %
                      self.conn.in_waiting)

            time.sleep(0.01)

    def get_exception(self):
        return self.exception
