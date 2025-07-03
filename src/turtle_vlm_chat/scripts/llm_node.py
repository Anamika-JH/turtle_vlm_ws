#!/usr/bin/env python3

import rospy
import re
import json
import random
import sys
import os
import rospkg
import yaml
import math, tf
sys.path.append(os.path.dirname(__file__))
from std_msgs.msg import String
from tf import TransformListener
from llm_interface import LLMInterface

CARDINALS = [
    "East", "North-East", "North", "North-West",
    "West", "South-West", "South", "South-East"
]


def _yaw_to_cardinal(yaw_deg: float) -> str:
    idx = int((yaw_deg + 22.5) // 45) % 8
    return CARDINALS[idx]

# ----------- Naturalization Helper -----------
def naturalize_response(raw_response: str) -> str:
    # Clean SYSTEM prefix
    cleaned = raw_response.replace("### SYSTEM MESSAGE:", "").strip()

    # Extract Action lines
    actions = re.findall(r'Action\s*\d+:\s*(.+)', cleaned)
    if not actions:
        # Try fallback "**ACTION:** <command>"
        if "**ACTION:**" in cleaned:
            fallback = cleaned.split("**ACTION:**")[-1].strip().split("\n")[0]
            actions = [fallback]
        else:
            return cleaned

    # Natural starter phrases
    starters = [
        "Sure, I’ll", "Alright, I'm going to", "Okay, proceeding to",
        "Understood! Now I’ll", "Got it, I’ll", "Cool, I’ll"
    ]

    natural_phrases = []
    for act in actions:
        act_lower = act.lower()
        if "move in a circle" in act_lower:
            natural_phrases.append("move in a circle")
        elif "turn left" in act_lower:
            degrees = re.search(r'(\d+)', act)
            natural_phrases.append(f"turn left {degrees.group(1)}°" if degrees else "turn left")
        elif "turn right" in act_lower:
            degrees = re.search(r'(\d+)', act)
            natural_phrases.append(f"turn right {degrees.group(1)}°" if degrees else "turn right")
        elif "navigate to" in act_lower:
            dest = re.search(r'navigate to (.+)', act, re.IGNORECASE)
            natural_phrases.append(f"go to the {dest.group(1).strip()}" if dest else "navigate to the goal")
        elif "stop" in act_lower:
            natural_phrases.append("stop")
        elif "report coordinates" in act_lower:
            natural_phrases.append("report my current position")
        else:
            natural_phrases.append(act.strip("."))

    if not natural_phrases:
        return "Alright, I’ll carry out your request."

    joined = ", then ".join(natural_phrases)
    prefix = random.choice(starters)
    return f"{prefix} {joined}."

# ----------- LLM Node Class -----------
class LLMNode:
    def __init__(self):
        rospy.init_node("llm_node")

        rospack   = rospkg.RosPack()
        yaml_path = os.path.join(
            rospack.get_path("turtle_vlm_chat"),
            "config",
            "destination.yaml"
        )
        with open(yaml_path) as f:
            dest_yaml = yaml.safe_load(f)

        # dest_yaml["destinations"] is the dict from the file
        self.llm = LLMInterface(destinations=dest_yaml["destinations"])
        self.listener = TransformListener()
        self.cmd_pub = rospy.Publisher("/llm_command", String, queue_size=10)
        self.chat_pub = rospy.Publisher("/chat_response", String, queue_size=10)

        rospy.Subscriber("/chat_input", String, self.on_input)

        rospy.loginfo("LLM Node ready and listening to /chat_input")
        rospy.spin()

    def on_input(self, msg):
        text = msg.data.strip()
        position = self.get_position()
        output = self.llm.process_input(text, **position)

        if output["type"] == "ACTIONS":
            # Publish structured command to executor
            self.cmd_pub.publish(String(data=json.dumps(output["content"])))
            # Also publish pretty version to GUI
            pretty_response = naturalize_response(output["content"])
            self.chat_pub.publish(String(data=pretty_response))
        else:
            pretty_response = naturalize_response(output["content"])
            self.chat_pub.publish(String(data=pretty_response))

    def get_position(self):
        try:
            self.listener.waitForTransform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(0.5)
            )
            (trans, rot) = self.listener.lookupTransform(
                "map", "base_footprint", rospy.Time(0)
            )
            _, _, yaw = tf.transformations.euler_from_quaternion(rot)
            yaw_deg = math.degrees(yaw)

            return {
                "position_x": f"{trans[0]:.3f}",
                "position_y": f"{trans[1]:.3f}",
                "position_z": f"{trans[2]:.3f}",
                "current_yaw": f"{yaw_deg:.1f}",
                "cardinal_direction": _yaw_to_cardinal(yaw_deg)
            }

        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return {
                "position_x": "unknown",
                "position_y": "unknown",
                "position_z": "unknown",
                "current_yaw": "unknown",
                "cardinal_direction": "unknown"
            }

if __name__ == "__main__":
    try:
        LLMNode()
    except rospy.ROSInterruptException:
        pass

