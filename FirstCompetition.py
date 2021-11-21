from Vision import Vision
from SerialConnection import SerialConnection
import queue

recieve_queue = queue.Queue()
send_queue = queue.Queue()

vision = Vision(send_queue, recieve_queue)
conn = SerialConnection('COM8', 9600, recieve_queue, send_queue)
conn.daemon = True
conn.start()

vision.run()

# clear up all the io buffers and close port
conn.conn.flush()
conn.conn.reset_input_buffer()
conn.conn.close()
