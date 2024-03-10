import os.path as osp

import numpy as np
import cv2

import svgparser

import pytest
from pytest import approx

DEBUG = True
TDIR = osp.dirname(__file__)
getfile = lambda *x: osp.join(TDIR, *x)

expected = {
    "eggplant": (
        "m 57.557452,262.05019 c 0,0 -28.544752,-29.0127 -6.083308,-37.90369 22.461445,-8.89099 23.865285,14.50635 23.865285,14.50635 0,0 14.506349,-56.62156 16.378136,-59.89719 1.871787,-3.27562 18.249925,-33.69216 21.057605,3.74358 2.80768,37.43574 -17.314031,61.30102 -17.314031,61.30102 0,0 33.224221,-30.88448 25.269131,-2.80768 -7.9551,28.07681 -32.288331,27.60887 -32.288331,27.60887",
        2,
        [
            (57.56, 262.05),
            (51.47, 224.15),
            (75.34, 238.65),
            (91.72, 178.76),
            (112.78, 182.50),
            (95.46, 243.8),
            (120.73, 240.99),
            (88.44, 268.6),
        ],
        "eggplant.png",
    ),
    "blob": (
        "M 118.40826,98.854517 C 104.74185,95.762321 90.525411,94.186715 76.540003,92.689884 72.37803,92.244434 57.032612,90.969407 52.456488,91.71652 c -7.077501,1.155494 -22.898332,10.21362 -27.788672,14.60046 -2.426853,2.17698 -7.193181,9.37029 -8.151344,12.97817 -4.17183,15.70879 4.883097,31.60244 15.932173,43.47691 7.137846,7.67107 20.106038,11.63509 30.011763,15.24937 22.014577,8.03241 44.908572,14.4378 68.545382,17.52055 16.29188,2.1248 16.93411,1.62227 32.97589,1.62227 15.05883,0 20.01998,0.37618 29.27075,-11.35592 5.18281,-6.57301 4.30051,-16.15993 4.44618,-23.68519 0.11623,-6.00413 0.35399,-14.57687 -1.48207,-20.44062 -4.37409,-13.96942 -17.30232,-23.70524 -27.04764,-34.71663 -4.37698,-4.94561 -5.02271,-6.8286 -10.37444,-10.707006 -4.86632,-3.526639 -20.56952,-11.127756 -28.5297,-9.733637 -2.41231,0.422486 -5.25282,7.519897 -6.66928,8.760274 -1.80197,1.577947 -3.07554,2.64442 -5.18722,3.568996 z",
        3,
        [
            (130.27, 86.53),
            (158.80, 96.26),
            (169.17, 106.97),
            (196.22, 141.68),
            (197.70, 162.12),
            (193.25, 185.81),
            (163.98, 197.16),
            (131.01, 195.54),
            (62.46, 178.02),
            (32.45, 162.77),
            (16.52, 119.30),
            (24.67, 106.32),
            (52.46, 91.72),
            (76.54, 92.69),
            (118.41, 98.86),
            (123.60, 95.29),
        ],
        "blob.png",
    ),
    "text": (
        "m -118.56674,427.38115 0.49637,-4.27875 3.32793,0.38608 q 7.53875,0.87455 9.37849,-0.15099 1.91552,-1.08556 2.38826,-5.16056 l 8.501354,-73.28217 q 0.464856,-4.00709 -1.159156,-5.43444 -1.616143,-1.49526 -9.086968,-2.36194 l -3.32792,-0.38607 0.50424,-4.34668 q 4.84274,0.97478 20.531495,2.79481 15.552926,1.80428 20.490202,1.96406 l -0.504253,4.34668 -3.327919,-0.38607 q -7.538753,-0.87456 -9.454287,0.211 -1.839738,1.02553 -2.304594,5.03262 l -3.774003,32.53211 41.565022,4.82191 3.774003,-32.53211 q 0.464856,-4.00709 -1.159157,-5.43445 -1.54822,-1.48738 -9.086972,-2.36195 l -3.32792,-0.38606 0.504252,-4.34669 q 4.84273,0.97478 20.531488,2.79482 15.552925,1.80427 20.4902,1.96406 l -0.504253,4.34668 -3.327918,-0.38607 q -7.538753,-0.87456 -9.454287,0.211 -1.839737,1.02553 -2.304593,5.03262 l -8.501357,73.28216 q -0.464856,4.00708 1.151279,5.50235 1.616133,1.49528 9.08697,2.36197 l 3.327918,0.38606 -0.496371,4.27875 q -4.842728,-0.97478 -20.531487,-2.79481 -15.552923,-1.80428 -20.490201,-1.96406 l 0.496371,-4.27875 3.327921,0.38607 q 7.538751,0.87455 9.378491,-0.15098 1.915534,-1.08557 2.38827,-5.16057 l 4.223101,-36.40337 -41.565021,-4.8219 -4.223102,36.40337 q -0.464856,4.00709 1.083362,5.49447 1.616134,1.49529 9.154887,2.36984 l 3.327919,0.38608 -0.496371,4.27875 q -4.842729,-0.9748 -20.531488,-2.79483 -15.552927,-1.80427 -20.490197,-1.96405 z"
        " m 104.320054,12.10204 0.496371,-4.27875 3.327918,0.38606 q 7.5387547,0.87457 9.3784934,-0.15098 1.9155338,-1.08555 2.3882692,-5.16057 l 8.4540835,-72.87466 q 0.4648559,-4.00709 -1.1591577,-5.43444 -1.616136,-1.49526 -9.08697343,-2.36194 L -3.7756,349.22184 l 0.5042523,-4.34669 78.3079567,9.08441 0.239071,31.75872 -3.463754,-0.40183 q -0.029,-8.05656 -0.791529,-12.75668 -0.686735,-4.76015 -3.333282,-8.64637 -2.57863,-3.87835 -7.37897,-5.81186 -4.724548,-1.99352 -12.670801,-2.91535 l -17.930008,-2.08003 q -4.754172,-0.55154 -5.891105,0.34903 -1.129056,0.83265 -1.546637,4.43223 l -3.868551,33.34713 12.496673,1.44971 q 9.576255,1.11094 12.678989,-1.90183 3.102737,-3.01276 4.213663,-12.58901 l 3.53167,0.4097 -4.270376,36.81087 -3.531669,-0.40971 q 1.095171,-9.44042 -1.227552,-13.15142 -2.322721,-3.71099 -11.898977,-4.82191 l -12.496672,-1.44973 -4.30977,37.15046 q -0.417583,3.59958 0.482983,4.73651 0.976362,1.07691 5.662615,1.62056 l 18.541259,2.15094 q 5.97667,0.69335 10.410008,0.45051 4.509133,-0.30288 7.785993,-1.84999 3.284741,-1.61503 5.582248,-3.61992 2.365421,-1.99701 4.256528,-5.83865 1.966903,-3.90166 3.215243,-7.54252 1.316259,-3.633 2.976195,-9.6352 l 3.463752,0.40183 -10.000907,35.18245 z"
        " m 95.083376,11.03048 0.496371,-4.27873 q 5.025833,0.58303 8.328224,0.002 3.310277,-0.64844 4.677585,-2.34825 1.443114,-1.75985 1.856592,-2.95082 0.421363,-1.25891 0.665609,-3.36433 l 8.068019,-69.54674 q 0.46486,-4.00709 -1.15914,-5.43445 -1.54823,-1.48738 -9.08698,-2.36195 l -3.327921,-0.38606 0.504253,-4.34669 23.499178,2.72611 q 2.51293,0.29152 3.13694,0.84573 0.63188,0.48629 1.39852,2.77781 l 21.11788,81.39875 39.51109,-74.77824 q 0.90766,-1.8908 1.5583,-2.15948 0.71856,-0.26079 3.36731,0.0465 l 23.49917,2.7261 -0.50425,4.34669 -3.32791,-0.38608 q -7.53876,-0.87455 -9.45428,0.211 -1.83975,1.02554 -2.30461,5.03263 l -8.50135,73.28215 q -0.46486,4.00709 1.08337,5.49448 1.61613,1.49529 9.15488,2.36984 l 3.32791,0.38607 -0.49636,4.27874 q -5.11441,-1.00629 -19.85233,-2.71602 -14.87374,-1.72548 -20.0827,-1.91678 l 0.49637,-4.27875 3.39583,0.39395 q 7.53876,0.87456 9.3785,-0.15099 1.84763,-1.09343 2.32037,-5.16844 l 9.25772,-79.80216 -0.13582,-0.0158 -44.37321,84.05698 q -1.2581,2.53846 -2.82017,2.35724 -1.69793,-0.19697 -2.43005,-3.37928 l -23.51309,-90.41813 -0.13584,-0.0158 -8.69833,74.98009 q -0.24425,2.10541 -0.12228,3.42736 0.12985,1.254 1.06381,3.28962 1.00978,1.97556 4.08369,3.36462 3.08176,1.32116 8.1076,1.9042 l -0.49636,4.27874 q -14.554835,-2.10147 -16.252749,-2.29844 -1.630001,-0.1891 -16.279361,-1.47557 z"
        " m 126.80047,14.70997 0.49638,-4.27875 3.32791,0.38606 q 7.53877,0.87457 9.3785,-0.15098 1.91555,-1.08555 2.38828,-5.16056 l 8.50137,-73.28216 q 0.46485,-4.00709 -1.15918,-5.43445 -1.61613,-1.49525 -9.08697,-2.36195 l -3.32792,-0.38606 0.50425,-4.34668 q 4.84274,0.97478 21.07483,2.85785 18.33751,2.1273 23.41062,2.30285 l -0.50426,4.34668 -4.61835,-0.53577 q -6.38415,-0.74062 -9.39911,-0.26442 -3.01496,0.4762 -3.62432,1.57564 -0.6015,1.03151 -0.92453,3.8161 l -8.60378,74.16509 q -0.41759,3.59958 0.483,4.73651 0.97635,1.07691 5.6626,1.62056 l 11.41001,1.32366 q 7.06333,0.81941 12.40134,-0.6951 5.33799,-1.5145 8.36194,-3.8481 3.02395,-2.33359 5.36495,-7.08667 2.41678,-4.81311 3.33042,-7.94215 0.98157,-3.12117 2.18672,-8.76315 l 3.46375,0.40182 -8.03131,35.41094 z"
        " m 95.59348,-36.67888 q 2.4661,-21.25794 17.76705,-34.488 15.37673,-13.29011 34.32548,-11.09189 19.08461,2.21398 30.94246,18.66353 11.92577,16.45742 9.46754,37.64746 -2.43458,20.98627 -17.7798,34.00471 -15.34524,13.01844 -34.36191,10.81233 -18.745,-2.17458 -30.77808,-18.3003 -12.03308,-16.12574 -9.58274,-37.24784 z"
        " m 14.5589,-0.30713 q -1.49699,12.90417 0.40893,23.17457 1.91381,10.20249 6.18138,16.13518 4.33549,5.94058 9.33358,9.13598 4.99811,3.19539 10.43145,3.8257 5.2975,0.61456 10.75069,-1.29955 5.52111,-1.90626 11.09358,-6.6286 5.57248,-4.72234 9.84697,-14.27575 4.27449,-9.55341 5.79512,-22.66134 1.18184,-10.18752 -0.046,-18.58964 -1.21998,-8.47006 -4.04067,-13.82193 -2.82066,-5.35187 -6.85707,-9.12401 -3.96847,-3.76425 -7.96957,-5.46735 -3.9253,-1.76317 -7.79657,-2.21227 -4.88999,-0.56728 -10.23587,1.01514 -5.34588,1.58242 -11.06696,5.8057 -5.64526,4.16327 -10.00043,13.22549 -4.3473,8.99432 -5.82854,21.76268 z",
        86,  # XXX didn't count
        [],
        "text.png",
    ),
}


@pytest.mark.parametrize("subject", list(expected.keys()))
def test_cubic(subject):
    path, n_cmds, keypts, filename = expected[subject]

    cmds = svgparser.parse_path(path)
    assert len(cmds) == n_cmds

    segs = svgparser.discretize_path(cmds, 0.1)
    seg_pts = np.array([pt for seg in segs for pt in seg.pts]).round(2)
    assert all(kpt in seg_pts for kpt in keypts)

    ## Now test overall appearance
    mask = draw_segments(segs)
    img = cv2.imread(getfile(filename), cv2.IMREAD_GRAYSCALE)

    if DEBUG:
        print(f"shapes mask={mask.shape}, expected={img.shape}")
        cv2.imshow("rendered", mask)
        cv2.imshow("expected", img)
        cv2.waitKey(0)

    assert mask.shape == img.shape, "Rendering has wrong shape"

    img_dil = cv2.dilate(img, np.ones((3, 3))) > 0
    mask_dil = mask > 0  # cv2.dilate(mask, np.ones((3, 3))) > 0

    if DEBUG:
        vis = np.zeros((*img.shape, 3), dtype=np.uint8)
        vis[img_dil] = (255, 255, 0)
        vis[mask > 0] = (40, 40, 255)
        cv2.imshow("rendering (red) contained in expected (blueish)?", vis)

        vis = np.zeros((*img.shape, 3), dtype=np.uint8)
        vis[mask_dil] = (40, 40, 255)
        vis[img > 0] = (255, 255, 0)
        cv2.imshow("rendering (red) covering all of expected (blueish)?", vis)

        cv2.waitKey(0)

    # Is our rendering contained within the expected?
    num_matched = np.count_nonzero(img_dil[mask > 0])
    num_predicted = np.count_nonzero(mask)
    assert num_matched / num_predicted == approx(1.0, abs=0.01)

    # Is our rendering complete?
    num_expected = np.count_nonzero(img)
    num_found = np.count_nonzero(mask_dil[img > 0])
    assert num_found / num_expected == approx(1.0, abs=0.01)


def draw_segments(segs):
    segs = segs.copy()

    seg_pts = np.array([pt for seg in segs if seg.drawing for pt in seg.pts]).round(2)
    dims = (np.max(seg_pts, axis=0) - np.min(seg_pts, axis=0)).round().astype(int)
    mask = np.zeros(dims[::-1])

    origin = np.min(seg_pts, axis=0)
    for seg in segs:
        if not seg.drawing:
            continue

        for startp, endp in zip(seg.pts[:-1], seg.pts[1:]):
            startp = np.array(startp) - origin
            endp = np.array(endp) - origin
            # Make line slightly thicker to account for different antialiasing
            cv2.line(mask, startp.astype(int), endp.astype(int), (255, 255, 255), 2)

    return mask
