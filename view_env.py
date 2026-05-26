"""
View the Go2 standing on the ground in the MuJoCo 3D viewer.
Usage:
    .venv/bin/python3 view_env.py
"""
import mujoco
import mujoco.viewer
from envs.go2_joystick import _find_xml

xml_path = _find_xml().replace("go2_mjx.xml", "scene_mjx.xml")
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# Reset to the XML "home" keyframe (standing pose, correct ctrl)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

print("Controls: left-drag=rotate  scroll=zoom  right-drag=pan  Space=pause  Esc=quit")

# launch() runs the sim + viewer together — simplest, no manual loop needed
mujoco.viewer.launch(mj_model, mj_data)
