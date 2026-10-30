import os, sys, traceback
REPO = "/home/zh/Desktop/EasyMimic/_cleanroom/easymimic_repro"
sys.path.insert(0, os.path.join(REPO, "src"))
import mujoco
from ee_retarget import ik_jaka_zu3 as ik

ABS = os.path.join(REPO, "third_party", "jaka-zu3-mujoco", "jaka_zu3_mujoco.xml")
REL = "third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml"
os.chdir(REPO)
print("cwd:", os.getcwd(), "| DEFAULT_XML_PATH isabs:", os.path.isabs(ik.DEFAULT_XML_PATH))
print("DEFAULT_XML_PATH:", ik.DEFAULT_XML_PATH)

print("\n--- 1) constructor with RELATIVE path ---")
try:
    ik.JakaZu3IK(REL)
    print("NO EXCEPTION (unexpected)")
except Exception as e:
    print("RAISED", type(e).__name__, "|", e)

print("\n--- 2) mujoco direct load with RELATIVE path ---")
try:
    m = mujoco.MjModel.from_xml_path(REL)
    print("LOADED OK nq=", m.nq)
except Exception as e:
    print("RAISED", type(e).__name__, "|", e)

print("\n--- 3) mujoco direct load with ABSOLUTE path ---")
try:
    m = mujoco.MjModel.from_xml_path(ABS)
    print("LOADED OK nq=", m.nq, "nu=", m.nu)
except Exception as e:
    print("RAISED", type(e).__name__, "|", e)

print("\n--- 4) EASYMIMIC_JAKA_XML relative override ---")
os.environ["EASYMIMIC_JAKA_XML"] = REL
import importlib
importlib.reload(ik)
print("reloaded DEFAULT_XML_PATH:", ik.DEFAULT_XML_PATH, "| isabs:", os.path.isabs(ik.DEFAULT_XML_PATH))
try:
    ik.JakaZu3IK()
    print("NO EXCEPTION (unexpected)")
except Exception as e:
    print("RAISED", type(e).__name__, "|", e)
