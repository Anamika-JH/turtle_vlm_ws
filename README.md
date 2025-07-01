# Turtle VLM Workspace – ROS + YOLO + LLM

A complete **ROS Noetic** workspace that turns a (simulated or real) TurtleBot3 Burger with an Intel RealSense depth camera into a conversational robot:

* **Perception:** YOLO v8 + MiDaS depth for 3-D object poses, published via a simple service (`/get_seen_objects`).
* **Reasoning / Language:** lightweight LLM front-end that converts natural-language commands into a structured **action dictionary**.
* **Control & Navigation:** turns these actions into `move_base` goals or direct `cmd_vel` twists.
* **Chat GUI:** Tkinter interface with speech I/O and image previews.
* **Gazebo world:** ready-to-run CPS lab environment with RViz configs.

<p align="center">
  <img src="https://raw.githubusercontent.com/Anamika-JH/turtle_vlm_ws/main/docs/preview.gif" width="650">
</p>

---

## ️📂 Directory layout

turtle_vlm_ws/
├── .catkin_workspace
├── src/
│ ├── turtle_vlm_chat/ # 🔑 main package (code + configs)
│ ├── turtle_vlm_gazebo/ # Gazebo world, RViz config, launch files
│ ├── turtle_vlm_perception/ # placeholder for future C++/CUDA nodes
│ └── turtlebot3_description_custom/
└── frames.{gv,pdf} # TF tree snapshots
