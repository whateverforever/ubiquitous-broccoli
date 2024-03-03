import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import check_grad, approx_fprime
from scipy.integrate import quad

source = "m 13.758333,135.99583 c 0,0 17.991667,-47.095831 29.633333,-21.16666 11.641667,25.92916 16.404167,35.98333 29.633333,15.875 13.229167,-20.10834 30.691671,1.05833 30.691671,1.05833 l 30.16249,-29.63333 7.40834,30.16249 24.87083,-21.69583"
penis = "m 57.557452,262.05019 c 0,0 -28.544752,-29.0127 -6.083308,-37.90369 22.461445,-8.89099 23.865285,14.50635 23.865285,14.50635 0,0 14.506349,-56.62156 16.378136,-59.89719 1.871787,-3.27562 18.249925,-33.69216 21.057605,3.74358 2.80768,37.43574 -17.314031,61.30102 -17.314031,61.30102 0,0 33.224221,-30.88448 25.269131,-2.80768 -7.9551,28.07681 -32.288331,27.60887 -32.288331,27.60887"
weird = "m 29.88877,106.18379 c 0.786548,0.31462 75.82309,37.28231 85.57627,9.59587 9.75318,-27.686442 22.02331,-59.148305 -15.888241,-37.28231 -37.911545,21.866 -67.643007,35.55191 -37.754237,54.90095 29.888772,19.34905 59.934848,47.97935 81.643538,22.18062 21.70869,-25.79873 38.54078,-39.48464 12.42744,-31.77649 -26.11335,7.70816 -60.878711,4.56197 -25.48411,19.66367 35.39459,15.1017 48.45127,23.28178 49.39512,8.18008 0.94386,-15.10169 11.64089,-38.54078 -6.92161,-16.83209 -18.5625,21.70868 -38.38347,21.07945 -17.46133,29.73146 20.92214,8.65201 34.92267,3.93273 38.69809,-1.41579 3.77543,-5.34851 -0.94385,-36.33845 -0.94385,-37.12499 0,-0.78655 -12.74206,-41.215047 -13.21399,-41.844284 -0.47193,-0.629237 -27.52913,-11.168962 -29.25953,-11.955507 -1.7304,-0.786548 -41.84428,-11.7982 -42.78814,-11.7982 -0.94385,0 -42.158893,7.865465 -42.78813,8.180083 -0.629238,0.314619 -39.484641,19.978286 -39.484641,19.978286 L 13.84322,101.93644 c 0,0 5.505826,20.45021 6.764301,20.92214 1.258475,0.47193 9.438559,5.97775 9.595867,-4.24735"
pencil = "m 30.34297,102.46427 c -8.96321,27.54839 14.694175,44.21679 39.012712,49.0805 5.083894,1.01678 15.22024,4.0604 21.394068,2.51695 2.960978,-0.74025 12.47923,-42.8409 7.550843,-45.30509 -6.860937,-3.43046 -9.613744,23.96285 -6.292369,28.94492 3.536379,5.30456 5.239959,3.72972 11.326266,6.29237 13.24589,5.57722 67.19498,28.29577 69.2161,0 0.29939,-4.19133 1.01725,-15.58415 -1.25847,-20.13559 -2.28909,-4.57818 -9.80743,5.38106 -10.0678,6.29237 -5.74396,20.10388 -6.99199,48.38089 8.80932,64.1822"
blob = "M 117.98199,88.093218 C 106.37732,85.094761 94.305619,83.56692 82.430085,82.115466 78.895997,81.68352 65.865631,80.447146 61.979873,81.17161 c -6.00977,1.120465 -19.44383,9.903986 -23.596399,14.157838 -2.060731,2.110991 -6.107999,9.086222 -6.921611,12.584742 -3.542456,15.23256 4.14642,30.6444 13.528603,42.1589 6.061012,7.43852 17.072788,11.28236 25.484109,14.78708 18.693399,7.78891 38.133555,14.00012 58.204445,16.98941 13.83405,2.06039 14.37939,1.57309 28.00106,1.57309 12.78701,0 16.99972,0.36478 24.85488,-11.01166 4.40092,-6.37375 3.65174,-15.67004 3.77542,-22.96716 0.0987,-5.82212 0.30059,-14.13498 -1.25848,-19.82097 -3.7142,-13.54593 -14.69204,-22.98662 -22.96716,-33.664195 -3.71665,-4.795681 -4.26497,-6.621576 -8.80932,-10.382414 -4.13217,-3.419729 -17.46635,-10.790417 -24.22563,-9.43856 -2.04839,0.409678 -4.46037,7.29193 -5.66314,8.494704 -1.53011,1.530112 -2.61156,2.564255 -4.40466,3.460803 z"
source = penis

rex_cmd = r"[a-zA-Z]\s(-?\d+\.?\d*,-?\d+\.?\d*\s*)*"


@dataclass
class PathCommand:
    cmd: str
    coords: Sequence[Tuple[float, float]]


def bezier_quad(start, end, control_pt1):
    p0 = np.array(start)
    p1 = np.array(control_pt1)
    p2 = np.array(end)

    assert len(p0) == 2
    assert len(p1) == 2
    assert len(p2) == 2

    def _spline_time(t):
        return (
            p1 + (1-t)**2*(p0-p1)+t**2*(p2-p1)
        )

    def _spline_dt(t):
        return np.reshape(
            2*(1-t)*(p1-p0)+2*t*(p2-p1),
            (-1,1)
        )

    # sanity check that wikipedia gradients are correct
    err = check_grad(_spline_time, _spline_dt, 0)
    assert err < 1e-5, f"err={err:.2f}"
    err = check_grad(_spline_time, _spline_dt, 0.5)
    assert err < 1e-5, f"err={err:.2f}"
    err = check_grad(_spline_time, _spline_dt, 1)
    assert err < 1e-5, f"err={err:.2f}"

    # For arc-length parameterization, we build a reverse lookup table to
    # interpolate from, to get t(s)
    ts = []
    ss = []
    for t in np.linspace(0, 1, 100):
        s, err = quad(lambda x: np.linalg.norm(_spline_dt(x)), 0, t)
        assert err < 1e-5, f"integration got large error {err}"
        ss.append(s)
        ts.append(t)

    def _spline_arc(s):
        t = np.interp(s, ss, ts)
        return _spline_time(t)

    arc_length = ss[-1]
    return arc_length, _spline_arc


def bezier_cubic(start, end, control_pt1, control_pt2):
    p0 = np.array(start)
    p1 = np.array(control_pt1)
    p2 = np.array(control_pt2)
    p3 = np.array(end)

    assert len(p0) == 2
    assert len(p1) == 2
    assert len(p2) == 2
    assert len(p3) == 2

    def _spline_time(t):
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def _spline_dt(t):
        return np.reshape(
            3 * (1 - t) ** 2 * (p1 - p0)
            + 6 * (1 - t) * t * (p2 - p1)
            + 3 * t**2 * (p3 - p2),
            (-1, 1),
        )

    # sanity check that wikipedia gradients are correct
    err = check_grad(_spline_time, _spline_dt, 0)
    assert err < 1e-5, f"err={err:.2f}"
    err = check_grad(_spline_time, _spline_dt, 0.5)
    assert err < 1e-5, f"err={err:.2f}"
    err = check_grad(_spline_time, _spline_dt, 1)
    assert err < 1e-5, f"err={err:.2f}"

    # For arc-length parameterization, we build a reverse lookup table to
    # interpolate from, to get t(s)
    ts = []
    ss = []
    for t in np.linspace(0, 1, 100):
        s, err = quad(lambda x: np.linalg.norm(_spline_dt(x)), 0, t)
        assert err < 1e-5, f"integration got large error {err}"
        ss.append(s)
        ts.append(t)

    def _spline_arc(s):
        t = np.interp(s, ss, ts)
        return _spline_time(t)

    arc_length = ss[-1]
    return arc_length, _spline_arc


def discretize_path(cmds: Sequence[PathCommand], spline_step=None):
    cursor = [0.0, 0.0]
    segments = []
    start_subpath = cursor

    for cmd in cmds:
        print("Executing command", cmd)

        if cmd.cmd == "m":
            # relative move
            # assert len(cmd.coords) == 1
            first = True
            for coords in cmd.coords:
                cursor[0] += coords[0]
                cursor[1] += coords[1]
                if first:
                    start_subpath = cursor.copy()
                    first = False
        elif cmd.cmd == "M":
            # absolute move
            assert len(cmd.coords) == 1
            cursor[0] = cmd.coords[0][0]
            cursor[1] = cmd.coords[0][1]
            start_subpath = cursor.copy()
        elif cmd.cmd == "z":
            segments.append([cursor.copy(), start_subpath.copy()])
            cursor = start_subpath.copy()
        elif cmd.cmd in ["c", "C"]:
            # cubic bezier
            assert (
                len(cmd.coords) % 3 == 0
            ), "Expected cubic path to have multiple of 3 coordinates!"

            is_relative = cmd.cmd == "c"
            for i in range(0, len(cmd.coords) - 2, 3):
                prevpt = np.array(cursor) if is_relative else np.array([0, 0])
                tipa = prevpt + cmd.coords[i]
                tipb = prevpt + cmd.coords[i + 1]
                nextpt = prevpt + cmd.coords[i + 2]

                alen, b = bezier_cubic(np.array(cursor), nextpt, tipa, tipb)
                if not spline_step:
                    spline_step = alen / 20

                asses = np.linspace(0, alen, int(alen // spline_step))
                if not len(asses):
                    # how does this happen?
                    continue
                pts = np.array([b(s) for s in asses])
                # plt.plot(pts[:, 0], pts[:, 1], "-x")
                segments.append(pts)
                cursor = nextpt
        elif cmd.cmd in ["q", "Q"]:
            # quad bezier
            assert (
                len(cmd.coords) % 2 == 0
            ), "Expected cubic path to have multiple of 2 coordinates!"

            is_relative = cmd.cmd.islower()
            for i in range(0, len(cmd.coords) - 1, 2):
                prevpt = np.array(cursor) if is_relative else np.array([0, 0])
                tipa = prevpt + cmd.coords[i]
                nextpt = prevpt + cmd.coords[i + 1]

                alen, b = bezier_quad(np.array(cursor), nextpt, tipa)
                if not spline_step:
                    spline_step = alen / 20

                asses = np.linspace(0, alen, int(alen // spline_step))
                if not len(asses):
                    # how does this happen?
                    continue
                pts = np.array([b(s) for s in asses])
                # plt.plot(pts[:, 0], pts[:, 1], "-x")
                segments.append(pts)
                cursor = nextpt
        elif cmd.cmd == "l":
            # relative polyline
            xs = [cursor[0]]
            ys = [cursor[1]]
            for coord in cmd.coords:
                cursor[0] += coord[0]
                cursor[1] += coord[1]
                xs.append(cursor[0])
                ys.append(cursor[1])
            # plt.plot(xs, ys)
            segments.append(list(zip(xs, ys)))
        elif cmd.cmd == "L":
            # absolute polyline
            xs = [cursor[0]]
            ys = [cursor[1]]
            for coord in cmd.coords:
                xs.append(coord[0])
                ys.append(coord[1])
            # plt.plot(xs, ys)
            segments.append(list(zip(xs, ys)))
        else:
            raise NotImplementedError(f"Unknown command {cmd.cmd}")

    return segments


def parse_path(source):
    cmds: List[PathCommand] = []
    for match in re.finditer(rex_cmd, source):
        thing = match.group(0)
        cmd, *coords = thing.split()
        coords = [tuple(float(comp) for comp in pair.split(",")) for pair in coords]
        print("building command", cmd, coords)
        cmds.append(PathCommand(cmd=cmd, coords=coords))
    return cmds

def main():
    cmds = parse_path(source)
    segments = discretize_path(cmds)
    for seg in segments:
        seg = np.array(seg)
        plt.plot(seg[:, 0], seg[:, 1], "-x")
    plt.axis("scaled")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()


