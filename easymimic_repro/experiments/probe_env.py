import importlib
for mod in ["numpy","scipy","mujoco","mink","qpsolvers","cv2","imageio","pandas","pyarrow","torch"]:
    try:
        m = importlib.import_module(mod)
        print("%-12s OK   %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as e:
        print("%-12s MISSING (%s)" % (mod, type(e).__name__))
