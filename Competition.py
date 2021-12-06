from Vision import Vision
from Connection import Connection
import queue


# Create two queues
from_arduino = queue.Queue()
to_arduino = queue.Queue()


# Initialise vision and start connection thread
vision = Vision(from_arduino, to_arduino)
conn = Connection(from_arduino, to_arduino)
conn.daemon = True
conn.start()

# Start main thread
vision.run()
