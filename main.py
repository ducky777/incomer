import time
import zmq

import numpy as np

from collections import deque

from cnn_model import CNNModel

lookbacks = 60

def receive_message():
    message = socket.recv()
    message = message.decode("utf-8")
    return message

def receive_new_data(data):
    message = receive_message()
    if message == "Connected!":
        print("Client reconnected!")
        socket.send(b"Confirmed reconnection")
        return
    message_type, value = message.split(";")
    data.append(float(value))
    socket.send(b"Received!")
    return data, message_type

def send_signal(prediction):
    print("Trying to send signal...")
    message = socket.recv()
    signal = b"%.3f" % prediction
    socket.send(signal)

if __name__ == '__main__':
    print("Starting app...")

    print('Loading model...')
    model = CNNModel('./best_EURUSD_saved.h5', x_max=0.02579)
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
        op, _ = receive_new_data(op)
        hi, _ = receive_new_data(hi)
        print("Sending high...")
        lo, _ = receive_new_data(lo)
        cl, _ = receive_new_data(cl)
        # print("Received Data")
        print("Open: %.5f, High: %.5f, Low: %.5f, Close: %.5f" % (op[-1], hi[-1], lo[-1], cl[-1]))
        # print(len(op))
        if len(op) == lookbacks:
            if start_signal:
                print("Sending signal...")
                pr = model.predict(op, hi, lo, cl)
                send_signal(pr)
            else:
                start_signal = True
