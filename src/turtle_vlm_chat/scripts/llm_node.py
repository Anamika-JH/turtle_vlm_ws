#!/usr/bin/env python3

import rospy
import re
import json
import random
import sys
import os
sys.path.append(os.path.dirname(__file__))
from std_msgs.msg import String
from tf import TransformListener
from llm_interface import LLMInterface

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

        self.llm = LLMInterface(destinations={
            "kitchen": {}, "secretary office": {}, "meeting room": {}
        })
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
            self.listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(2.0))
            (trans, _) = self.listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            return {
                "position_x": str(trans[0]),
                "position_y": str(trans[1]),
                "position_z": str(trans[2]),
                "current_yaw": "0",  # You can compute actual yaw if needed
                "cardinal_direction": "unknown"
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

