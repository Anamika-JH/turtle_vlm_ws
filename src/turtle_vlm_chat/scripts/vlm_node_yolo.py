#!/usr/bin/env python3

import rospy
import rospkg
import os
import torch
import tf
import time
import yaml
import math
import cv2
import numpy as np

from ultralytics import YOLO              
from std_srvs.srv import Empty, EmptyResponse
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Pose, Quaternion
from cv_bridge import CvBridge
from filterpy.kalman import KalmanFilter
from scipy.spatial import ConvexHull
from scipy import ndimage

import message_filters
from turtle_vlm_chat.srv import SeenObjects, SeenObjectsResponse
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String


# ---------------------------------------------------------------------------

def to_builtin_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin_type(v) for v in obj]
    else:
        return obj


class ImageManager:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.last_image_time = None
        self.fx = self.fy = self.cx = self.cy = None
        self.intrinsics_ready = False

        info_topic = rospy.get_param("topics/camera_info", "/camera/camera_info")
        rospy.Subscriber(info_topic,
                         CameraInfo,
                         self.camera_info_callback,
                         queue_size=1)


    def camera_info_callback(self, msg):
        K = msg.K               # 3×3 row-major camera matrix
        self.fx, self.fy = K[0], K[4]
        self.cx, self.cy = K[2], K[5]
        self.intrinsics_ready = True
        rospy.loginfo_once(
            f"[CameraInfo] fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}")
    
    def get_intrinsics(self):
        if not self.intrinsics_ready:
            raise RuntimeError("Camera intrinsics not received yet")
        return self.fx, self.fy, self.cx, self.cy
    
    def _depth_to_metres(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":       
            depth = depth.astype(np.float32) * 0.001
        # 32FC1 is already metres
        return depth
    
    def image_callback(self, rgb_msg, depth_msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            self.last_image_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr(f"[ImageManager] Failed to convert images: {e}")


class DepthEstimator:
    def __init__(self, get_intrinsics_fn, mono_model, mono_transform, device):
        self.get_intrinsics_fn = get_intrinsics_fn
        self.mono_model   = mono_model
        self.mono_transform = mono_transform
        self.device       = device

    def pixel_to_3d_point(self, u, v, depth):
        if depth <= 0.0 or not np.isfinite(depth):
            return None
        fx, fy, cx, cy = self.get_intrinsics_fn()
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return np.array([x, y, depth])

    def get_valid_depth_value(self, depth_image, x, y, radius=5):
        h, w = depth_image.shape
        xs = slice(max(0, x - radius), min(w, x + radius + 1))
        ys = slice(max(0, y - radius), min(h, y + radius + 1))
        region = depth_image[ys, xs]
        vals = region[np.isfinite(region) & (region > 0)]
        return float(np.median(vals)) if vals.size else None

    def monocular_depth_estimate(self, rgb_image, mask):
        h, w = mask.shape
        img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        input_batch = self.mono_transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.mono_model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_map = depth_map / np.max(depth_map)
        depth_map *= 4.0  # max depth in metres
        return float(np.median(depth_map[mask]))


class ObjectTracker:
    def __init__(self, initial_pose, process_noise=1e-4):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.eye(6)
        self.kf.F[:3, 3:] = np.eye(3)
        self.kf.H = np.eye(3, 6)
        self.kf.R = np.eye(3) * 0.1
        self.kf.P[3:, 3:] *= 1e3
        self.kf.Q[3:, 3:] = process_noise
        self.kf.x[:3] = initial_pose.reshape(3, 1)

    def update(self, measurement):
        self.kf.predict()
        self.kf.update(np.array(measurement).reshape(3, 1))

class PerceptionModule:
    def __init__(self, data_logger=None):
        self.data_logger = data_logger
        self.listener = tf.TransformListener()
        self.image_manager = ImageManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trackers = {}
        # Load YOLO model
        yolo_ckpt = rospy.get_param("models/yolo_checkpoint", "yolov8x.pt")
        self.yolo_model = YOLO(yolo_ckpt)

        # Load MiDaS for fallback depth
        self.mono_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
        self.mono_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        self.depth_estimator = DepthEstimator(
            get_intrinsics_fn=self.image_manager.get_intrinsics,
            mono_model=self.mono_model,
            mono_transform=self.mono_transform,
            device=self.device
        )
        self.base_frame = rospy.get_param("perception/base_frame", "map")
        self.camera_frame = rospy.get_param("perception/camera_frame", "realsense_link_optical")
        self.seen_objects = {}
        self.image_publisher = rospy.Publisher("/llm_image_output", Image, queue_size=10)
        self.overlay_publisher = rospy.Publisher("/perception/yolo_overlay", Image, queue_size=1)
        rospy.Subscriber(
            "/perception/request_image",
            String,
            lambda _msg: self.send_latest_image(),
            queue_size=1
        )
        
        self.setup_subscribers()

        self.detection_confidence_threshold = rospy.get_param("/perception/detection_confidence_threshold", 0.55)
        self.seen_objects_service = rospy.Service("/get_seen_objects", SeenObjects, self.handle_seen_objects_service)
        self.timer = rospy.Timer(rospy.Duration(1.5), self.periodic_detection_callback)
        rospy.Service('/clear_seen_objects', Empty, self._srv_clear)
    
    def _srv_clear(self, req):
        self.seen_objects.clear()
        rospy.loginfo("[Perception] seen_objects cleared.")
        return EmptyResponse()
    
    def setup_subscribers(self):
        rgb_topic = rospy.get_param("topics/camera_color", "/camera/image_raw")
        depth_topic = rospy.get_param("topics/camera_depth", "/camera/depth/image_raw")
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.image_manager.image_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def periodic_detection_callback(self, event):
        self.detect_objects()

    def detect_objects(self):
        """
        Run YOLO, triangulate depth, smooth each object’s 3-D position
        with a Kalman filter, and store / visualise the results.
        """
        if self.image_manager.rgb_image is None:
            rospy.logwarn("[YOLO-VLM] No RGB image for detection.")
            return []

        if not self.image_manager.intrinsics_ready:
            rospy.logwarn_once("[YOLO-VLM] Waiting for /camera/camera_info …")
            return []
        
        rgb       = self.image_manager.rgb_image.copy()
        dets      = self.yolo_model(rgb, verbose=False)[0]      # first batch
        depth_img = self.image_manager.depth_image

        accepted, vis_boxes, vis_labels = [], [], []

        for det in dets.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cls_id  = int(det.cls)
            conf    = float(det.conf)

            if conf < self.detection_confidence_threshold:
                continue

            label = self.yolo_model.names[cls_id]

            BLACKLIST      = {"potted plant", "cup", "umbrella",
                            "sign stop", "light traffic", "car", "sink", "vase", "airplane"}
            REMAP_TO_TRASH = {"potted plant"}

            if label in BLACKLIST:
                continue
            if label in REMAP_TO_TRASH:
                label = "trash can"

            # ---------- 1. depth → camera xyz ----------
            if depth_img is None:
                rospy.logwarn_once("[YOLO-VLM] Depth stream not yet available.")
                continue

            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
            depth_val = self.depth_estimator.get_valid_depth_value(depth_img, cx, cy, radius=7)

            if depth_val is None:
                box_patch = depth_img[max(0, y1):y2, max(0, x1):x2]
                vals = box_patch[np.isfinite(box_patch) & (box_patch > 0)]
                depth_val = float(np.median(vals)) if vals.size else None

            if depth_val is None:          # MiDaS fallback
                mask          = np.zeros(depth_img.shape, np.uint8)
                mask[y1:y2, x1:x2] = 1
                depth_val     = self.depth_estimator.monocular_depth_estimate(rgb, mask)

            if depth_val is None:
                continue

            cam_xyz = self.depth_estimator.pixel_to_3d_point(cx, cy, depth_val)
            if cam_xyz is None:
                continue

            # ---------- 2. camera xyz → <base_frame> ----------
            try:
                # use the RGB stamp for TF accuracy
                now = rospy.Time.now()
                self.listener.waitForTransform(self.base_frame,
                                            self.camera_frame,
                                            now,
                                            rospy.Duration(0.5))
                pt_cam  = PointStamped(header=Header(stamp=now,
                                                    frame_id=self.camera_frame),
                                    point=Point(*cam_xyz))
                pt_base = self.listener.transformPoint(self.base_frame, pt_cam)

                raw_pos = np.array([pt_base.point.x,
                                    pt_base.point.y,
                                    pt_base.point.z])

                # ---------- 2½.  Kalman smoothing ----------
                key = label.lower()
                if not hasattr(self, "trackers"):
                    self.trackers = {}                     # first call: create dict
                if key not in self.trackers:
                    self.trackers[key] = ObjectTracker(raw_pos)
                else:
                    self.trackers[key].update(raw_pos)
                    raw_pos = self.trackers[key].kf.x[:3]  # use filtered value

                # PoseStamped for downstream use
                pose_msg = PoseStamped(
                    header=Header(stamp=now, frame_id=self.base_frame),
                    pose  = Pose(position=Point(*raw_pos),
                                orientation=Quaternion(w=1.0))
                )

                # ---------- 3. store in memory ----------
                self.seen_objects.setdefault(key, []).append(
                    {"pose": {"x": float(raw_pos[0]),
                            "y": float(raw_pos[1]),
                            "z": float(raw_pos[2])},
                    "confidence": conf,
                    "timestamp": now.to_sec()}
                )

                # ---------- 4. overlay ----------
                vis_boxes.append([x1, y1, x2, y2])
                vis_labels.append(f"{label} {conf:.2f}")
                accepted.append({"label": label,
                                "confidence": conf,
                                "pose": pose_msg})

            except (tf.Exception, tf.LookupException,
                    tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                rospy.logwarn("[YOLO-VLM] TF failed for %s: %s", label, tf_err)

        self.publish_overlay(rgb, vis_boxes, vis_labels)
        return accepted


    def publish_overlay(self, image, boxes, labels):
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        ros_image = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
        self.overlay_publisher.publish(ros_image)
    
    def handle_seen_objects_service(self, req):
        """
        Return the internal object-memory (last poses & confidences)
        as a YAML string – used by llm_to_goal_node.py.
        """
        safe_memory = to_builtin_type(self.seen_objects)          # numpy → native
        return SeenObjectsResponse(memory_json=yaml.dump(safe_memory))

    def get_seen_objects(self, max_age_sec: float = 10.0):
        """
        Convenience helper: filter the object memory by age.
        """
        now = rospy.Time.now().to_sec()
        recent = {}
        for label, entries in self.seen_objects.items():
            keep = [e for e in entries if now - e["timestamp"] <= max_age_sec]
            if keep:
                recent[label] = keep
        return recent

    # ----------  depth / TF helper that is still used by YOLO  -------------
    def pixel_to_base_frame(self, u, v, depth_val):
        """
        Project a pixel (u,v,depth) into <base_frame>.  Returns geometry_msgs/Point.
        """
        try:
            cam_xyz = self.depth_estimator.pixel_to_3d_point(u, v, depth_val)
            if cam_xyz is None:
                return None
            now = rospy.Time.now()
            self.listener.waitForTransform(self.base_frame,
                                           self.camera_frame,
                                           now,
                                           rospy.Duration(0.5))
            pt_cam = PointStamped(
                header=Header(stamp=now, frame_id=self.camera_frame),
                point=Point(*cam_xyz)
            )
            pt_base = self.listener.transformPoint(self.base_frame, pt_cam)
            return pt_base.point
        except Exception as e:
            rospy.logwarn(f"[YOLO-VLM] TF transform failed: {e}")
            return None

    # ----------  public helpers exposed to the rest of the robot stack -----
    def send_latest_image(self):
        if self.image_manager.rgb_image is None:
            rospy.logwarn("[YOLO-VLM] No RGB image to forward.")
            return
        msg = self.image_manager.bridge.cv2_to_imgmsg(self.image_manager.rgb_image,
                                                      encoding="bgr8")
        self.image_publisher.publish(msg)

    def get_object_locations(self):
        """
        Force a detection pass and return a dict[label] → Pose entries.
        """
        detections = self.detect_objects()        # populates self.seen_objects
        locs = {}
        for det in detections:
            p = det["pose"].pose.position
            locs.setdefault(det["label"].lower(), []).append(
                {"x": p.x, "y": p.y, "z": p.z, "confidence": det["confidence"]}
            )
        return locs

    def get_detected_objects(self):
        """
        Returns the list of unique labels in the most recent frame.
        """
        detections = self.detect_objects()
        return list({d["label"] for d in detections})

    def get_object_pose(self, object_name: str, prob_thresh: float = 0.25):
        """
        Look for an object by label and return a PoseStamped in <base_frame>.
        """
        detections = self.detect_objects()
        for d in detections:
            if d["label"].lower() == object_name.lower() and d["confidence"] >= prob_thresh:
                return d["pose"]
        rospy.logwarn(f"[YOLO-VLM] '{object_name}' not in view (>{prob_thresh}).")
        return None

    def get_angle_to_object(self, object_name: str):
        """
        Compute bearing from robot to object (deg, robot frame).
        """
        pose = self.get_object_pose(object_name)
        if pose is None or not hasattr(self, "robot_pose"):
            return None

        obj_x, obj_y = pose.pose.position.x, pose.pose.position.y
        q = self.robot_pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        robot_yaw_deg = math.degrees(yaw)
        angle_deg = math.degrees(math.atan2(obj_y, obj_x)) - robot_yaw_deg
        return angle_deg


def load_yaml_config_to_param_server():
    rospack = rospkg.RosPack()
    config_path = os.path.join(rospack.get_path("turtle_vlm_chat"), "config", "perception_config.yaml")
    if not os.path.exists(config_path):
        rospy.logerr(f"[Config] YAML config not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        rospy.set_param(f"{key}/{sub_key}", sub_val)
                else:
                    rospy.set_param(key, value)
            rospy.loginfo(f"[Config] Loaded perception_config.yaml successfully from {config_path}")
        except yaml.YAMLError as exc:
            rospy.logerr(f"[Config] Error parsing YAML file: {exc}")
if __name__ == "__main__":
    rospy.init_node("vlm_node_yolo")
    load_yaml_config_to_param_server()
    perception = PerceptionModule()
    rospy.loginfo("[Perception] Service /get_seen_objects is ready.")
    rospy.spin() 
