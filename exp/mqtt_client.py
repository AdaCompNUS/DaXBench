import logging
import pickle
import time

import paho.mqtt.client as mqtt


class Client:
    def __init__(
        self, logger_name, host="crane1.d2.comp.nus.edu.sg", port=1883, topics=None
    ):
        topics = [] if topics is None else topics
        self.logger = logging.getLogger(logger_name)
        self._connected = False

        # connect to socket
        self.client = mqtt.Client()
        self.client.connect(host, port, 60)
        self.logger.debug("{} node: connected".format(logger_name))

        # set up msg queue
        self.in_msg_queue = {"exit": []}
        for topic in topics:
            self.in_msg_queue[topic] = []

        def on_connect(client, userdata, flags, rc):
            self._connected = True
            self.logger.info("Connected with result code " + str(rc))
            for topic in topics:
                client.subscribe(topic)

        def on_message(client, userdata, msg):
            try:
                msg = pickle.loads(msg.payload, encoding="latin-1")
            except TypeError:
                msg = pickle.loads(msg.payload)
            self.in_msg_queue[msg["topic"]].append(msg)
            self.logger.debug("{} node: received: {}".format(logger_name, msg["topic"]))

        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.loop_start()

        t0 = time.time()
        while not self._connected:
            time.sleep(0.1)
            if time.time() - t0 > 10:
                raise RuntimeError("mqtt client connection timeout!")

    def send_msg(self, msg):
        data_string = pickle.dumps(msg, protocol=2)
        self.client.publish(msg["topic"], data_string)
        self.logger.debug("send: {}".format({"topic": msg["topic"]}))

    def await_msg(self, topic, timeout=30.0):
        # block the thread and wait for msg
        assert topic
        t0 = time.time()

        while True:
            if len(self.in_msg_queue["exit"]) > 0:
                self.logger.info("exit")
                exit(0)
            if len(self.in_msg_queue[topic]) > 0:
                msg = self.in_msg_queue[topic].pop(-1)
                self.in_msg_queue[topic] = []
                self.logger.debug("received: {}".format(msg))
                return msg
            if time.time() - t0 > timeout:
                raise RuntimeError("mqtt client await topic [%s] timeout!" % topic)
            time.sleep(0.1)


if __name__ == "__main__":
    c = Client("test", topics=["test"])
    c.await_msg("test", timeout=2)
