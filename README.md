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

### Directory layout
```text
turtle_vlm_ws/
├── .catkin_workspace
├── src/
│   ├── turtle_vlm_chat/             # Main package (code + configs)
│   ├── turtle_vlm_gazebo/           # Gazebo world, RViz config, launch files
│   └── turtlebot3_description_custom/  # Custom URDF for TurtleBot3 + Realsense
└── frames.gv
```

### Key scripts

| Path | Purpose |
|------|---------|
| `turtle_vlm_chat/scripts/vlm_node_yolo.py` | Runs YOLO v8, estimates 3-D poses, maintains “seen objects” memory |
| `turtle_vlm_chat/scripts/llm_to_goal_node.py` | Converts parsed LLM actions into real robot motion |
| `turtle_vlm_chat/scripts/llm_node.py` | Thin wrapper that posts prompts to your chosen LLM endpoint (OpenAI, Ollama, …) |
| `turtle_vlm_chat/scripts/chatGUI_SR.py` | Desktop GUI with speech recognition (Google) & TTS (gTTS / pyttsx3) |

---

## Quick-start (simulation)

> Tested on **Ubuntu 20.04 + ROS Noetic + Python 3.8**.

```bash
# 1. clone (already done if you're reading this from your repo clone)
git clone git@github.com:Anamika-JH/turtle_vlm_ws.git
cd turtle_vlm_ws

# 2. download YOLO + CLIP + SAM checkpoints (~500 MB total)
src/turtle_vlm_chat/models/download_models.sh

# 3. install missing Python deps (virtualenv recommended)
pip install -r src/turtle_vlm_chat/requirements.txt

# 4. build workspace
catkin_make
source devel/setup.bash

# 5. launch Gazebo world + navigation stack
roslaunch turtle_vlm_gazebo nav_sim.launch

# 6. start perception
rosrun turtle_vlm_chat vlm_node_yolo.py

# 7. start LLM interface (edit API key env vars first!)
rosrun turtle_vlm_chat llm_node.py

# 8. start action executor
rosrun turtle_vlm_chat llm_to_goal_node.py

# 9. optional: start the chat GUI
rosrun turtle_vlm_chat chatGUI_SR.py
```
## License

This repository is released under the **[MIT License](LICENSE)**.

All third-party model checkpoints keep their original licenses:

- **YOLO v8** – © Ultralytics  
- **CLIP** – © OpenAI  
- **SAM** – © Meta AI  
- **MiDaS** – © Intel ISL
