import os

REPO = "/home/zh/Desktop/EasyMimic/_cleanroom/easymimic_repro"
src = os.path.join(REPO, "experiments", "t1_verify.py")
code = open(src).read()

code = code.replace(
    'HERE = os.path.dirname(os.path.abspath(__file__))',
    'HERE = os.path.join(%r, "experiments")' % REPO)
code = code.replace(
    'z0 = float(d.xpos[bid_cube][2])',
    'z0 = float(d.xpos[bid_cube][2])\ncube_xy_close = d.xpos[bid_cube][:2].copy()')
code = code.replace(
    'cube_pos = d.xpos[bid_cube].copy()',
    'cube_pos = d.xpos[bid_cube].copy()\n'
    'slip = float(((cube_pos[:2] - cube_xy_close) ** 2).sum() ** 0.5)\n'
    'print("  SLIP_PROBE cube xy at close=%s after lift=%s |dxy|=%.6f m = %.2f mm"\n'
    '      % (np.array2string(cube_xy_close, precision=6), np.array2string(cube_pos[:2], precision=6), slip, 1000.0 * slip))')

exec(compile(code, src, "exec"), {"__name__": "__probe__", "__file__": src})
