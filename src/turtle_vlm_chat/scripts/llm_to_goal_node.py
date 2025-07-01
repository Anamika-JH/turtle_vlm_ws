#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
import tf.transformations
import math
from tf import TransformListener
from typing import Optional
import time
import sys
import os
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from turtle_vlm_chat.srv import SeenObjects
import yaml
import json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from commands_parser import CommandParser
from destination_resolver import DestinationResolver

class LLMToGoalNode:
    def __init__(self):
        rospy.init_node("llm_to_goal_node")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.cmd_pub = rospy.Publisher("/chat_response", String, queue_size=10)
        self.img_req_pub  = rospy.Publisher("/perception/request_image", String, queue_size=1, latch=True)
        self.scan_req_pub = rospy.Publisher("/perception/force_detect", String, queue_size=1, latch=True)
        self.should_stop = False  
        self.seen_objects = {}
        self.tf_listener = TransformListener()
        self.parser = CommandParser(action_executor=self)
        yaml_path = os.path.join(os.path.dirname(__file__), "../config/destination.yaml")
        self.destination_resolver = DestinationResolver(yaml_path)
        rospy.Subscriber("/llm_command", String, self.command_callback)

        rospy.loginfo("LLM to Goal Node Ready")
        rospy.spin()

    def command_callback(self, msg):
        raw_command = msg.data.strip()
        rospy.loginfo(f"Received command: {raw_command}")
        if "what objects have you seen" in raw_command.lower():
            self.query_seen_objects()
            return
        try:
            content = json.loads(raw_command)
        except json.JSONDecodeError:
            content = f"Action 1: {raw_command}"

        parsed = self.parser.parse_input({"type": "ACTIONS", "content": content})
        rospy.loginfo(f"Parsed: {parsed}")
        if parsed['type'] == 'ACTIONS':
            for action in parsed['content']:
                self.execute_action(action)
        elif parsed['type'] == 'RESPONSE':
            self.cmd_pub.publish(String(data=parsed['content']))

    def execute_action(self, action):
        act = action.get("action")
        action_type = act
        action_args = action

        if act == "STOP":
            self.should_stop = True
            self.stop_motion()
            self.move_base_client.cancel_all_goals()
            rospy.loginfo("Navigation goal canceled.")
            self.cmd_pub.publish(String(data="Emergency STOP received. All motion halted."))
            return
        else:
            self.should_stop = False

        if act == "FORWARD":
            self.move_straight(action.get('distance', 1.0))
        elif act == "BACKWARD":
            self.move_straight(-action.get('distance', 1.0))
        elif act == "TURN_LEFT":
            self.rotate_degrees(action.get('angle', 90.0))
        elif act == "TURN_RIGHT":
            self.rotate_degrees(-action.get('angle', 90.0))
        elif act == "ROTATE":
            self.rotate_degrees(action.get('angle', 360.0))
        elif act == "WAIT":
            rospy.loginfo(f"Waiting for {action.get('duration', 1.0)} seconds")
            time.sleep(action.get('duration', 1.0))
        elif act == "CIRCULAR_MOTION":
            self.move_in_circle(
                radius=action.get('radius', 1.0),
                speed=action.get('speed', 0.2),
                angle=action.get('angle', 360.0)
            )
        elif act == "DESCRIBE_SURROUNDINGS":
            self.describe_surroundings()
        elif act == "LIST_SEEN_OBJECTS":
            self.list_seen_objects()
        elif act == "LOOK_AROUND":
            self.trigger_perception_scan()
        elif act == "FIND":
            object_name = action_args.get("object_name")
            self.search_for_object(object_name)
        elif act == "REPORT_OBJECT_POSITION":
            object_name = action_args.get("object_name")
            self.report_object_position(object_name)
        elif act == "REPORT_OBJECT_LOCATIONS":
            self.report_object_locations()
        elif act == "MOVE_TO_OBJECT":
            object_name = action.get("object") or action.get("object_name")
            if not object_name:
                self.speak_and_respond("I’m not sure which object you want me to approach.")
                return
            self.move_to_object(object_name)
        elif act == "REPORT_COORDINATES":
            self.report_position()
        elif act == "GO_TO_COORDINATES":
            coords = action.get("coordinates", {})
            x = coords.get("x", 0.0)
            y = coords.get("y", 0.0)
            z = coords.get("z", 0.0)
            self.publish_goal(x, y, 0.0)
        elif act == "NAVIGATE_AROUND_OBJECT":
            self.navigate_around_object(action_args.get("object_name"),
                                        action_args.get("clearance", 0.5))
        elif act == "NAVIGATE_TO_DESTINATION":
            dest_phrase = action.get("destination_name", "").lower()
            resolved_key, method = self.destination_resolver.resolve(dest_phrase)
            if resolved_key:
                coords = self.destination_resolver.coords[resolved_key]
                self.publish_goal(coords["x"], coords["y"], 0.0)
                rospy.loginfo(f"Resolved '{dest_phrase}' to '{resolved_key}' via {method}")
                self.cmd_pub.publish(String(data=f"Navigating to {resolved_key.replace('_', ' ').title()}"))
            else:
                self.cmd_pub.publish(String(data=f"Destination '{dest_phrase}' not recognized."))
        else:
            rospy.logwarn(f"Unsupported action: {act}")

    def publish_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]

        self.goal_pub.publish(goal)
        rospy.loginfo(f"Published goal: ({x}, {y}, yaw={yaw})")

    def move_straight(self, distance):
        if self.should_stop:
            rospy.loginfo("STOP flag set. Aborting move_straight.")
            return

        rospy.loginfo(f"Moving straight for {distance:.2f} meters")
        twist = Twist()
        speed = 0.2
        twist.linear.x = speed if distance > 0 else -speed
        duration = abs(distance) / speed

        rate = rospy.Rate(10)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration:
            if self.should_stop:
                rospy.loginfo("STOP flag set. Interrupting move_straight mid-motion.")
                break
            self.vel_pub.publish(twist)
            rate.sleep()

        self.stop_motion()

    def rotate_degrees(self, degrees):
        if self.should_stop:
            rospy.loginfo("STOP flag set. Aborting rotate_degrees.")
            return

        radians = math.radians(degrees)
        rospy.loginfo(f"Rotating {'clockwise' if degrees < 0 else 'counter-clockwise'} by {degrees} degrees")
        twist = Twist()
        angular_speed = 0.5  # rad/s
        twist.angular.z = angular_speed if degrees > 0 else -angular_speed
        duration = abs(radians) / angular_speed

        rate = rospy.Rate(10)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration:
            if self.should_stop:
                rospy.loginfo("STOP flag set. Interrupting rotate_degrees mid-motion.")
                break
            self.vel_pub.publish(twist)
            rate.sleep()

        self.stop_motion()

    def move_in_circle(self, radius=1.0, speed=0.2, angle=360.0):
        if self.should_stop:
            rospy.loginfo("STOP flag set. Aborting move_in_circle.")
            return

        rospy.loginfo(f"Executing circular motion: radius={radius}, angle={angle} degrees, speed={speed} m/s")

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = speed / radius  # omega = v / r

        arc_length = math.radians(angle) * radius
        duration = arc_length / speed

        rate = rospy.Rate(10)
        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < duration:
            if self.should_stop:
                rospy.loginfo("STOP flag set. Interrupting circular motion.")
                break
            self.vel_pub.publish(twist)
            rate.sleep()

        self.stop_motion()

    def report_position(self):
        try:
            self.tf_listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(4.0))
            (trans, _) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            msg = f"Current position: x = {trans[0]:.2f}, y = {trans[1]:.2f}"
            rospy.loginfo(msg)
            self.cmd_pub.publish(String(data=msg))
        except Exception as e:
            error_msg = f"Could not get current position: {e}"
            rospy.logwarn(error_msg)
            self.cmd_pub.publish(String(data=error_msg))

    def format_seen_objects_naturally(self, seen_dict):
        descriptions = []
        for label, entries in seen_dict.items():
            if not entries:
                continue
            latest = sorted(entries, key=lambda e: e['timestamp'])[-1]
            pos = latest.get("pose", {})
            confidence = latest.get("confidence", 0.0)

            # Skip low-confidence observations
            if confidence < 0.20:
                continue

            # Convert label like "chair_a black object" to "a black chair"
            parts = label.replace("_", " ").replace(" object", "").split()
            if len(parts) >= 2:
                object_name = parts[0]
                color = parts[1]
                readable = f"a {color} {object_name}"
            else:
                readable = label.replace("_", " ")

            description = f"I saw {readable} with {confidence:.2f} confidence at position " \
                          f"({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})."
            descriptions.append(description)

        if descriptions:
            return "Here's what I observed:\n" + "\n".join(descriptions)
        else:
            return "I haven't seen any confident objects recently."
    
    
    def query_seen_objects(self):
        rospy.wait_for_service('/get_seen_objects')
        try:
            get_seen = rospy.ServiceProxy('/get_seen_objects', SeenObjects)
            response = get_seen()
            if response.memory_json:
                seen_dict = yaml.safe_load(response.memory_json)
                if not seen_dict:
                    msg = "I haven't seen any objects recently."
                else:
                    msg = self.format_seen_objects_naturally(seen_dict)
            else:
                msg = "I haven't seen any objects recently."
            rospy.loginfo(msg)
            self.cmd_pub.publish(String(data=msg))
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to query seen objects: {e}")
            self.cmd_pub.publish(String(data="Could not retrieve seen objects."))
    
    def stop_motion(self):
        twist = Twist()
        self.vel_pub.publish(twist)
        rospy.loginfo("Motion stopped.")
        self.should_stop = False  

    # publish text both to /chat_response *and* to the ROS log
    def speak_and_respond(self, text: str):
        rospy.loginfo(text)
        self.cmd_pub.publish(String(data=text))


    # ----------  perception conveniences ------------------------------------------
    def _query_seen_service(self, max_age=10.0):
        """
        Convenience wrapper around /get_seen_objects.
        Returns a python dict[label] -> list[entries] or {} on failure.
        """
        rospy.wait_for_service('/get_seen_objects')
        try:
            srv = rospy.ServiceProxy('/get_seen_objects', SeenObjects)
            reply = srv()
            return yaml.safe_load(reply.memory_json) or {}
        except Exception as e:
            rospy.logwarn(f"[LLM→Goal] /get_seen_objects failed: {e}")
            return {}


    # ----------  chat-only commands -----------------------------------------------
    def send_image(self):
        try:
            # just reuse the latched publisher
            self.img_req_pub.publish(String(data="send_now"))
            self.speak_and_respond(
                "Here is the image of my current surroundings.")
        except Exception as e:
            rospy.logerr(f"[LLM→Goal] Failed to request image: {e}")
            self.speak_and_respond(
                "I could not send an image at the moment.")


    def describe_surroundings(self):
        """
        Turn the internal object memory into a short natural-language summary.
        """
        seen = self._query_seen_service(max_age=12.0)
        text = self.format_seen_objects_naturally(seen)
        self.speak_and_respond(text)


    def list_seen_objects(self):
        """
        A more concise variant – just enumerate labels.
        """
        seen = self._query_seen_service(max_age=60.0)
        if not seen:
            self.speak_and_respond("I haven't seen any objects recently.")
            return
        names = ", ".join(sorted(seen.keys()))
        self.speak_and_respond(f"I have recently seen: {names}.")


    def trigger_perception_scan(self):
        
        self.scan_req_pub.publish(String(data="scan"))
        self.speak_and_respond("Scanning my surroundings…")


    # ----------  object-centric navigation helpers --------------------------------
    def _lookup_object_pose(self, object_name: str) -> Optional[PoseStamped]:
        """
        Return the latest PoseStamped for `object_name` from the perception memory,
        or None if not found or too old.
        """
        seen = self._query_seen_service(max_age=30.0)
        for label, entries in seen.items():          # ← use the fresh dict
            if object_name.lower() in label.lower():
                latest = max(entries, key=lambda e: e['timestamp'])  # newest
                p = latest["pose"]
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map"
                pose.pose.position.x = p["x"]
                pose.pose.position.y = p["y"]
                pose.pose.position.z = p["z"]
                pose.pose.orientation.w = 1.0
                return pose
        return None


    def move_to_object(self, object_name):
        pose = self._lookup_object_pose(object_name)
        if pose is None:
            self.speak_and_respond(f"I cannot locate {object_name}.")
            return
        self.goal_pub.publish(pose)
        self.speak_and_respond(f"Moving towards the {object_name}.")


    def search_for_object(self, object_name):
        """
        Simple strategy: rotate 360° slowly, hoping the perception module
        will pick the object up.  Very naïve but works indoors.
        """
        self.speak_and_respond(f"Searching for {object_name}.")
        self.rotate_degrees(360)
        pose = self._lookup_object_pose(object_name)
        if pose:
            self.speak_and_respond(f"I found {object_name}.")
        else:
            self.speak_and_respond(f"I could not find {object_name}.")


    def report_object_position(self, object_name):
        pose = self._lookup_object_pose(object_name)
        if pose:
            x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
            self.speak_and_respond(f"The {object_name} is at x={x:.2f}, y={y:.2f}, z={z:.2f}.")
        else:
            self.speak_and_respond(f"I do not have a position for {object_name}.")

    def _refresh_seen_objects(self):
        try:
            get_seen = rospy.ServiceProxy('/get_seen_objects', SeenObjects)
            response = get_seen()
            if response.memory_json:
                self.seen_objects = yaml.safe_load(response.memory_json)
            else:
                self.seen_objects = {}
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to refresh seen objects: {e}")
            self.seen_objects = {}
    
    def report_object_locations(self):
        self._refresh_seen_objects()
        if not self.seen_objects:
            msg = "I have not detected any objects recently."
        else:
            lines = []
            for label, infos in self.seen_objects.items():
                for info in infos:
                    pose = info["pose"]
                    confidence = info["confidence"]
                    lines.append(
                        f"I saw {label} with {confidence:.2f} confidence at position "
                        f"({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})."
                    )
            msg = "\n".join(lines)
        rospy.loginfo(msg)
        self.cmd_pub.publish(String(data=msg))
    
    def navigate_around_object(self, object_name, clearance=0.5):
        """
        Drive a simple 4-point box around the object.  Requires `move_base`.
        """
        target = self._lookup_object_pose(object_name)
        if target is None:
            self.speak_and_respond(f"I cannot locate {object_name}.")
            return

        ox, oy = target.pose.position.x, target.pose.position.y
        waypoints = [(ox + clearance, oy),
                    (ox, oy + clearance),
                    (ox - clearance, oy),
                    (ox, oy - clearance)]

        self.speak_and_respond(f"Circumnavigating the {object_name}.")
        for wx, wy in waypoints:
            g = MoveBaseGoal()
            g.target_pose.header.frame_id = "map"
            g.target_pose.header.stamp = rospy.Time.now()
            g.target_pose.pose.position.x = wx
            g.target_pose.pose.position.y = wy
            g.target_pose.pose.orientation.w = 1.0
            self.move_base_client.send_goal(g)
            self.move_base_client.wait_for_result()
            if self.should_stop:
                self.speak_and_respond("Navigation interrupted.")
                return
        self.speak_and_respond(f"Finished moving around the {object_name}.")


    def rotate_to_face(self, object_name):
        pose = self._lookup_object_pose(object_name)
        if pose is None:
            self.speak_and_respond(f"I cannot see a {object_name}.")
            return
        dx = pose.pose.position.x
        dy = pose.pose.position.y
        yaw_to_obj = math.degrees(math.atan2(dy, dx))

        # robot yaw
        self.tf_listener.waitForTransform("map", "base_footprint",
                                        rospy.Time(0), rospy.Duration(1.0))
        (trans, rot) = self.tf_listener.lookupTransform("map",
                                                        "base_footprint",
                                                        rospy.Time(0))
        robot_yaw = math.degrees(tf.transformations.euler_from_quaternion(rot)[2])
        delta = yaw_to_obj - robot_yaw
        self.rotate_degrees(delta)
        self.speak_and_respond(f"Now facing the {object_name}.")


    # ----------  info helpers -----------------------------------------------------
    def get_cardinal_direction(self, yaw_deg: float) -> str:
        idx = int((yaw_deg + 22.5) // 45) % 8
        return ["East", "North-East", "North", "North-West",
                "West", "South-West", "South", "South-East"][idx]


    def get_coordinates(self) -> str:
        try:
            self.tf_listener.waitForTransform("map", "base_footprint",
                                            rospy.Time(0), rospy.Duration(0.5))
            trans, _ = self.tf_listener.lookupTransform("map", "base_footprint",
                                                        rospy.Time(0))
            return f"My current coordinates are x={trans[0]:.2f}, y={trans[1]:.2f}."
        except Exception:
            return "I cannot determine my current coordinates."


    def report_orientation(self):
        try:
            self.tf_listener.waitForTransform("map", "base_footprint",
                                            rospy.Time(0), rospy.Duration(0.5))
            _, rot = self.tf_listener.lookupTransform("map", "base_footprint",
                                                    rospy.Time(0))
            yaw_deg = math.degrees(tf.transformations.euler_from_quaternion(rot)[2])
            direction = self.get_cardinal_direction(yaw_deg)
            self.speak_and_respond(f"My orientation is {yaw_deg:.1f}°, facing {direction}. {self.get_coordinates()}")
        except Exception as e:
            rospy.logwarn(f"[LLM→Goal] Orientation unavailable: {e}")
            self.speak_and_respond("I cannot determine my orientation right now.")


if __name__ == "__main__":
    try:
        LLMToGoalNode()
    except rospy.ROSInterruptException:
        pass

