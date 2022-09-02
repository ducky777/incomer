import time
import zmq

import numpy as np

from collections import deque

from cnn_model import CNNModel

from absl import app
from absl import flags

lookbacks = 60

FLAGS = flags.FLAGS

flags.DEFINE_string('symbol', None, '(str) Symbol to be traded on')
flags.DEFINE_integer('timeframe', None, '(int) Time frame to trade on')
flags.DEFINE_integer('lookbacks', None, '(int) Number of lookbacks')
flags.DEFINE_string('model_filename', None, '(str) filename of model to use')

flags.mark_flag_as_required("symbol")
flags.mark_flag_as_required("timeframe")
flags.mark_flag_as_required("lookbacks")
flags.mark_flag_as_required("model_filename")

def receive_message(socket):
    message = socket.recv()
    message = message.decode("utf-8")
    return message

def receive_new_data(socket, data):
    message = receive_message(socket)
    if message == "Connected!":
        print("Client reconnected!")
        socket.send(b"Confirmed reconnection")
        return
    message_type, value = message.split(";")
    data.append(float(value))
    socket.send(b"Received!")
    return data, message_type

def send_signal(socket, prediction):
    message = socket.recv()
    signal = b"%.3f" % prediction
    socket.send(signal)

def main(argv):
    lookbacks = FLAGS.lookbacks - 1

    print("Starting app...")

    print('Loading model...')
    model_path = 'models/%s%i_%i' % \
        (FLAGS.symbol, FLAGS.timeframe, FLAGS.lookbacks)
    model = CNNModel(model_path, FLAGS.model_filename)
    print("Model successfully loaded")

    op = deque(maxlen=lookbacks)
    hi = deque(maxlen=lookbacks)
    lo = deque(maxlen=lookbacks)
    cl = deque(maxlen=lookbacks)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Waiting for connection...")
    start_signal = False
    while True:
        op, _ = receive_new_data(socket, op)
        hi, _ = receive_new_data(socket, hi)
        lo, _ = receive_new_data(socket, lo)
        cl, _ = receive_new_data(socket, cl)
        # print("Received Data")
        print("Open: %.5f, High: %.5f, Low: %.5f, Close: %.5f" % (op[-1], hi[-1], lo[-1], cl[-1]))
        # print(len(op))
        if len(op) == lookbacks:
            if start_signal:
                print("Sending signal...")
                pr = model.predict(op, hi, lo, cl)
                send_signal(socket, pr)
            else:
                start_signal = True

if __name__ == '__main__':
    app.run(main)
