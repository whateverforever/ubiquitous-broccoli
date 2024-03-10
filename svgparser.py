import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import check_grad, approx_fprime
from scipy.integrate import quad

source = "m 13.758333,135.99583 c 0,0 17.991667,-47.095831 29.633333,-21.16666 11.641667,25.92916 16.404167,35.98333 29.633333,15.875 13.229167,-20.10834 30.691671,1.05833 30.691671,1.05833 l 30.16249,-29.63333 7.40834,30.16249 24.87083,-21.69583"
penis = "m 57.557452,262.05019 c 0,0 -28.544752,-29.0127 -6.083308,-37.90369 22.461445,-8.89099 23.865285,14.50635 23.865285,14.50635 0,0 14.506349,-56.62156 16.378136,-59.89719 1.871787,-3.27562 18.249925,-33.69216 21.057605,3.74358 2.80768,37.43574 -17.314031,61.30102 -17.314031,61.30102 0,0 33.224221,-30.88448 25.269131,-2.80768 -7.9551,28.07681 -32.288331,27.60887 -32.288331,27.60887"
weird = "m 29.88877,106.18379 c 0.786548,0.31462 75.82309,37.28231 85.57627,9.59587 9.75318,-27.686442 22.02331,-59.148305 -15.888241,-37.28231 -37.911545,21.866 -67.643007,35.55191 -37.754237,54.90095 29.888772,19.34905 59.934848,47.97935 81.643538,22.18062 21.70869,-25.79873 38.54078,-39.48464 12.42744,-31.77649 -26.11335,7.70816 -60.878711,4.56197 -25.48411,19.66367 35.39459,15.1017 48.45127,23.28178 49.39512,8.18008 0.94386,-15.10169 11.64089,-38.54078 -6.92161,-16.83209 -18.5625,21.70868 -38.38347,21.07945 -17.46133,29.73146 20.92214,8.65201 34.92267,3.93273 38.69809,-1.41579 3.77543,-5.34851 -0.94385,-36.33845 -0.94385,-37.12499 0,-0.78655 -12.74206,-41.215047 -13.21399,-41.844284 -0.47193,-0.629237 -27.52913,-11.168962 -29.25953,-11.955507 -1.7304,-0.786548 -41.84428,-11.7982 -42.78814,-11.7982 -0.94385,0 -42.158893,7.865465 -42.78813,8.180083 -0.629238,0.314619 -39.484641,19.978286 -39.484641,19.978286 L 13.84322,101.93644 c 0,0 5.505826,20.45021 6.764301,20.92214 1.258475,0.47193 9.438559,5.97775 9.595867,-4.24735"
pencil = "m 30.34297,102.46427 c -8.96321,27.54839 14.694175,44.21679 39.012712,49.0805 5.083894,1.01678 15.22024,4.0604 21.394068,2.51695 2.960978,-0.74025 12.47923,-42.8409 7.550843,-45.30509 -6.860937,-3.43046 -9.613744,23.96285 -6.292369,28.94492 3.536379,5.30456 5.239959,3.72972 11.326266,6.29237 13.24589,5.57722 67.19498,28.29577 69.2161,0 0.29939,-4.19133 1.01725,-15.58415 -1.25847,-20.13559 -2.28909,-4.57818 -9.80743,5.38106 -10.0678,6.29237 -5.74396,20.10388 -6.99199,48.38089 8.80932,64.1822"
blob = "M 117.98199,88.093218 C 106.37732,85.094761 94.305619,83.56692 82.430085,82.115466 78.895997,81.68352 65.865631,80.447146 61.979873,81.17161 c -6.00977,1.120465 -19.44383,9.903986 -23.596399,14.157838 -2.060731,2.110991 -6.107999,9.086222 -6.921611,12.584742 -3.542456,15.23256 4.14642,30.6444 13.528603,42.1589 6.061012,7.43852 17.072788,11.28236 25.484109,14.78708 18.693399,7.78891 38.133555,14.00012 58.204445,16.98941 13.83405,2.06039 14.37939,1.57309 28.00106,1.57309 12.78701,0 16.99972,0.36478 24.85488,-11.01166 4.40092,-6.37375 3.65174,-15.67004 3.77542,-22.96716 0.0987,-5.82212 0.30059,-14.13498 -1.25848,-19.82097 -3.7142,-13.54593 -14.69204,-22.98662 -22.96716,-33.664195 -3.71665,-4.795681 -4.26497,-6.621576 -8.80932,-10.382414 -4.13217,-3.419729 -17.46635,-10.790417 -24.22563,-9.43856 -2.04839,0.409678 -4.46037,7.29193 -5.66314,8.494704 -1.53011,1.530112 -2.61156,2.564255 -4.40466,3.460803 z"
text = (
    "m -118.56674,427.38115 0.49637,-4.27875 3.32793,0.38608 q 7.53875,0.87455 9.37849,-0.15099 1.91552,-1.08556 2.38826,-5.16056 l 8.501354,-73.28217 q 0.464856,-4.00709 -1.159156,-5.43444 -1.616143,-1.49526 -9.086968,-2.36194 l -3.32792,-0.38607 0.50424,-4.34668 q 4.84274,0.97478 20.531495,2.79481 15.552926,1.80428 20.490202,1.96406 l -0.504253,4.34668 -3.327919,-0.38607 q -7.538753,-0.87456 -9.454287,0.211 -1.839738,1.02553 -2.304594,5.03262 l -3.774003,32.53211 41.565022,4.82191 3.774003,-32.53211 q 0.464856,-4.00709 -1.159157,-5.43445 -1.54822,-1.48738 -9.086972,-2.36195 l -3.32792,-0.38606 0.504252,-4.34669 q 4.84273,0.97478 20.531488,2.79482 15.552925,1.80427 20.4902,1.96406 l -0.504253,4.34668 -3.327918,-0.38607 q -7.538753,-0.87456 -9.454287,0.211 -1.839737,1.02553 -2.304593,5.03262 l -8.501357,73.28216 q -0.464856,4.00708 1.151279,5.50235 1.616133,1.49528 9.08697,2.36197 l 3.327918,0.38606 -0.496371,4.27875 q -4.842728,-0.97478 -20.531487,-2.79481 -15.552923,-1.80428 -20.490201,-1.96406 l 0.496371,-4.27875 3.327921,0.38607 q 7.538751,0.87455 9.378491,-0.15098 1.915534,-1.08557 2.38827,-5.16057 l 4.223101,-36.40337 -41.565021,-4.8219 -4.223102,36.40337 q -0.464856,4.00709 1.083362,5.49447 1.616134,1.49529 9.154887,2.36984 l 3.327919,0.38608 -0.496371,4.27875 q -4.842729,-0.9748 -20.531488,-2.79483 -15.552927,-1.80427 -20.490197,-1.96405 z"
    " m 104.320054,12.10204 0.496371,-4.27875 3.327918,0.38606 q 7.5387547,0.87457 9.3784934,-0.15098 1.9155338,-1.08555 2.3882692,-5.16057 l 8.4540835,-72.87466 q 0.4648559,-4.00709 -1.1591577,-5.43444 -1.616136,-1.49526 -9.08697343,-2.36194 L -3.7756,349.22184 l 0.5042523,-4.34669 78.3079567,9.08441 0.239071,31.75872 -3.463754,-0.40183 q -0.029,-8.05656 -0.791529,-12.75668 -0.686735,-4.76015 -3.333282,-8.64637 -2.57863,-3.87835 -7.37897,-5.81186 -4.724548,-1.99352 -12.670801,-2.91535 l -17.930008,-2.08003 q -4.754172,-0.55154 -5.891105,0.34903 -1.129056,0.83265 -1.546637,4.43223 l -3.868551,33.34713 12.496673,1.44971 q 9.576255,1.11094 12.678989,-1.90183 3.102737,-3.01276 4.213663,-12.58901 l 3.53167,0.4097 -4.270376,36.81087 -3.531669,-0.40971 q 1.095171,-9.44042 -1.227552,-13.15142 -2.322721,-3.71099 -11.898977,-4.82191 l -12.496672,-1.44973 -4.30977,37.15046 q -0.417583,3.59958 0.482983,4.73651 0.976362,1.07691 5.662615,1.62056 l 18.541259,2.15094 q 5.97667,0.69335 10.410008,0.45051 4.509133,-0.30288 7.785993,-1.84999 3.284741,-1.61503 5.582248,-3.61992 2.365421,-1.99701 4.256528,-5.83865 1.966903,-3.90166 3.215243,-7.54252 1.316259,-3.633 2.976195,-9.6352 l 3.463752,0.40183 -10.000907,35.18245 z"
    " m 95.083376,11.03048 0.496371,-4.27873 q 5.025833,0.58303 8.328224,0.002 3.310277,-0.64844 4.677585,-2.34825 1.443114,-1.75985 1.856592,-2.95082 0.421363,-1.25891 0.665609,-3.36433 l 8.068019,-69.54674 q 0.46486,-4.00709 -1.15914,-5.43445 -1.54823,-1.48738 -9.08698,-2.36195 l -3.327921,-0.38606 0.504253,-4.34669 23.499178,2.72611 q 2.51293,0.29152 3.13694,0.84573 0.63188,0.48629 1.39852,2.77781 l 21.11788,81.39875 39.51109,-74.77824 q 0.90766,-1.8908 1.5583,-2.15948 0.71856,-0.26079 3.36731,0.0465 l 23.49917,2.7261 -0.50425,4.34669 -3.32791,-0.38608 q -7.53876,-0.87455 -9.45428,0.211 -1.83975,1.02554 -2.30461,5.03263 l -8.50135,73.28215 q -0.46486,4.00709 1.08337,5.49448 1.61613,1.49529 9.15488,2.36984 l 3.32791,0.38607 -0.49636,4.27874 q -5.11441,-1.00629 -19.85233,-2.71602 -14.87374,-1.72548 -20.0827,-1.91678 l 0.49637,-4.27875 3.39583,0.39395 q 7.53876,0.87456 9.3785,-0.15099 1.84763,-1.09343 2.32037,-5.16844 l 9.25772,-79.80216 -0.13582,-0.0158 -44.37321,84.05698 q -1.2581,2.53846 -2.82017,2.35724 -1.69793,-0.19697 -2.43005,-3.37928 l -23.51309,-90.41813 -0.13584,-0.0158 -8.69833,74.98009 q -0.24425,2.10541 -0.12228,3.42736 0.12985,1.254 1.06381,3.28962 1.00978,1.97556 4.08369,3.36462 3.08176,1.32116 8.1076,1.9042 l -0.49636,4.27874 q -14.554835,-2.10147 -16.252749,-2.29844 -1.630001,-0.1891 -16.279361,-1.47557 z"
    " m 126.80047,14.70997 0.49638,-4.27875 3.32791,0.38606 q 7.53877,0.87457 9.3785,-0.15098 1.91555,-1.08555 2.38828,-5.16056 l 8.50137,-73.28216 q 0.46485,-4.00709 -1.15918,-5.43445 -1.61613,-1.49525 -9.08697,-2.36195 l -3.32792,-0.38606 0.50425,-4.34668 q 4.84274,0.97478 21.07483,2.85785 18.33751,2.1273 23.41062,2.30285 l -0.50426,4.34668 -4.61835,-0.53577 q -6.38415,-0.74062 -9.39911,-0.26442 -3.01496,0.4762 -3.62432,1.57564 -0.6015,1.03151 -0.92453,3.8161 l -8.60378,74.16509 q -0.41759,3.59958 0.483,4.73651 0.97635,1.07691 5.6626,1.62056 l 11.41001,1.32366 q 7.06333,0.81941 12.40134,-0.6951 5.33799,-1.5145 8.36194,-3.8481 3.02395,-2.33359 5.36495,-7.08667 2.41678,-4.81311 3.33042,-7.94215 0.98157,-3.12117 2.18672,-8.76315 l 3.46375,0.40182 -8.03131,35.41094 z"
    " m 95.59348,-36.67888 q 2.4661,-21.25794 17.76705,-34.488 15.37673,-13.29011 34.32548,-11.09189 19.08461,2.21398 30.94246,18.66353 11.92577,16.45742 9.46754,37.64746 -2.43458,20.98627 -17.7798,34.00471 -15.34524,13.01844 -34.36191,10.81233 -18.745,-2.17458 -30.77808,-18.3003 -12.03308,-16.12574 -9.58274,-37.24784 z"
    " m 14.5589,-0.30713 q -1.49699,12.90417 0.40893,23.17457 1.91381,10.20249 6.18138,16.13518 4.33549,5.94058 9.33358,9.13598 4.99811,3.19539 10.43145,3.8257 5.2975,0.61456 10.75069,-1.29955 5.52111,-1.90626 11.09358,-6.6286 5.57248,-4.72234 9.84697,-14.27575 4.27449,-9.55341 5.79512,-22.66134 1.18184,-10.18752 -0.046,-18.58964 -1.21998,-8.47006 -4.04067,-13.82193 -2.82066,-5.35187 -6.85707,-9.12401 -3.96847,-3.76425 -7.96957,-5.46735 -3.9253,-1.76317 -7.79657,-2.21227 -4.88999,-0.56728 -10.23587,1.01514 -5.34588,1.58242 -11.06696,5.8057 -5.64526,4.16327 -10.00043,13.22549 -4.3473,8.99432 -5.82854,21.76268 z"
)
source = text

rex_cmd = r"[a-zA-Z]\s(-?\d+\.?\d*,-?\d+\.?\d*\s*)*"

Point2 = Union[np.ndarray, Sequence[float]]


@dataclass
class PathCommand:
    cmd: str
    coords: Sequence[Tuple[float, float]]


@dataclass
class Segment:
    pts: Sequence[Point2]
    drawing: bool = True


def bezier_quad(start, end, control_pt1):
    p0 = np.array(start)
    p1 = np.array(control_pt1)
    p2 = np.array(end)

    assert len(p0) == 2
    assert len(p1) == 2
    assert len(p2) == 2

    def _spline_time(t):
        return p1 + (1 - t) ** 2 * (p0 - p1) + t**2 * (p2 - p1)

    def _spline_dt(t):
        return np.reshape(2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1), (-1, 1))

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
            cursor_old = cursor.copy()
            cursor[0] += cmd.coords[0][0]
            cursor[1] += cmd.coords[0][1]
            segments.append(Segment([cursor_old, cursor], drawing=False))
            start_subpath = cursor.copy()

            pts = [start_subpath]
            for coord in cmd.coords[1:]:
                print("processing", coord)
                cursor[0] += coord[0]
                cursor[1] += coord[1]
                pts.append(cursor.copy())

            # subsequent move commands behave like line commands
            if pts:
                print("points from move", pts)
                segments.append(Segment(pts))

            # raise RuntimeError("done")
        elif cmd.cmd == "M":
            # absolute move
            assert len(cmd.coords) == 1
            cursor_old = cursor.copy()
            cursor[0] = cmd.coords[0][0]
            cursor[1] = cmd.coords[0][1]
            segments.append(Segment([cursor_old, cursor_old], drawing=False))
            start_subpath = cursor.copy()
        elif cmd.cmd == "z":
            segments.append(Segment([cursor.copy(), start_subpath.copy()]))
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
                segments.append(Segment(pts))
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
                segments.append(Segment(pts))
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
            segments.append(Segment(list(zip(xs, ys))))
        elif cmd.cmd == "L":
            # absolute polyline
            xs = [cursor[0]]
            ys = [cursor[1]]
            for coord in cmd.coords:
                cursor[0] = coord[0]
                cursor[1] = coord[1]
                xs.append(cursor[0])
                ys.append(cursor[1])
            # plt.plot(xs, ys)
            segments.append(Segment(list(zip(xs, ys))))
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

    csv = []
    for seg in segments:
        pts = np.array(seg.pts)
        if not seg.drawing:
            csv.append(f"{','.join(str(w) for w in pts[-1].round(2))},false\n")
            continue

        for pt in pts[:-1]:
            csv.append(f"{','.join(str(w) for w in pt.round(2))},true\n")
        plt.plot(pts[:, 0], pts[:, 1], "-x", c="gray")

    plt.axis("scaled")
    plt.gca().invert_yaxis()
    plt.show()

    with open("out.csv", "w") as fh:
        fh.writelines(csv)


if __name__ == "__main__":
    main()
