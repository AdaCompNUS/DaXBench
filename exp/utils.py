#!/usr/bin/env python
import mqtt_client


def argparse_common(p):
    """Common between server and client.

    Args:
        p: ArgumentParse
    """
    p.add_argument(
        "-o",
        "--obs_topic",
        type=str,
        default="pax_exp_obs",
        help="Name of the observation topic.",
    )
    p.add_argument(
        "-a",
        "--action_topic",
        type=str,
        default="pax_exp_action",
        help="Name of the action topic.",
    )
    p.add_argument(
        "-m",
        "--mqtt",
        type=str,
        default="crane1.d2.comp.nus.edu.sg:1883",
        help="mqtt server address.",
    )
    return p


def create_mqtt_client(mqtt_server_url, topics):
    mqtt_host = mqtt_server_url.split(":")[0]
    mqtt_port = int(mqtt_server_url.split(":")[1])
    mqtt_interface = mqtt_client.Client(
        "pax_exp_server", host=mqtt_host, port=mqtt_port, topics=topics
    )
    return mqtt_interface
