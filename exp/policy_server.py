"""A simple policy server for Pax hardware experiment.

Event loop:
- Wait for observation.
- Once an observation is received, infer an action.
- Publish the inferred action.

Observation msg fields:
- topic: str, the topic name
- payload: Any, the observation

Action msg fields:
- topic: str, the topic name
- payload: Any, the action
"""
import argparse
from abc import ABC, abstractmethod

import cv2

import utils


class PolicyHandler(ABC):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def infer(self):
        pass


class MockPolicyHandler(PolicyHandler):
    def __init__(self):
        pass

    def init(self):
        pass

    def infer(self, obs):
        """Take heightmap as input."""
        try:
            cv2.imwrite("/tmp/mock_policy_handler_obs_dump.jpg", obs)
        except Exception:
            pass
        return {
            "camera_config": "",
            "primitive": "pick_place",
            "params": {
                "pose0": ([0.4, 0.29375, 0.024], [1, 0, 0, 0]),
                "pose1": ([0.40625, 0.19062500000000002, 0.024], [1, 0, 0, 0]),
            },
        }


def cmdline_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p = utils.argparse_common(p)
    p.add_argument("policy", type=str, choices=["mock", "shac"], help="Policy name.")

    return p.parse_args()


def create_policy_handler(policy_name):
    if policy_name == "mock":
        return MockPolicyHandler()


if __name__ == "__main__":
    args = cmdline_args()
    print("Loaded policy: ", args.policy)
    policy_handler = create_policy_handler(args.policy)
    policy_handler.init()
    mqtt_interface = utils.create_mqtt_client(args.mqtt, [args.obs_topic])

    print("Server ready. Entering the event loop.")
    while True:
        obs = mqtt_interface.await_msg(args.obs_topic, timeout=3600)["payload"]
        action = policy_handler.infer(obs)
        action_msg = {
            "topic": args.action_topic,
            "payload": action,
        }
        mqtt_interface.send_msg(action_msg)
