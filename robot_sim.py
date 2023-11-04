import sys
import math
import typing as t
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

import pygame

WIDTH = 1280
HEIGHT = 800
FPS = 60

pygame.init()
FramePerSec = pygame.time.Clock()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Arm Plotter")

    robot_xy = [WIDTH // 2, HEIGHT // 2]
    target = None

    rob = Robot(robot_xy, [200, 100])
    rob.plan_move_to([3.14, 3.14], [2 * np.pi / 10, 2 * np.pi / 5])

    while True:
        screen.fill((80, 80, 80))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("restarting")
                    rob.plan_move_to([0.0, 0], [2 * np.pi / 10, 2 * np.pi / 5])
                    rob.plan_move_to([3.14, 3.14], [2 * np.pi / 10, 2 * np.pi / 5])
                if event.key == pygame.K_b:
                    print("restarting backwards")
                    rob.plan_move_to([0.0, 0], [-2 * np.pi / 10, -2 * np.pi / 5])
                if event.key == pygame.K_w:
                    print("running random whatever")
                    sign_a = np.random.choice([-1, 1])
                    sign_b = np.random.choice([-1, 1])
                    rob.plan_move_to(
                        np.random.uniform(0, 2 * np.pi, size=2),
                        [sign_a * 2 * np.pi / 10, sign_b * 2 * np.pi],
                    )
                if event.key == pygame.K_i:
                    print("inverse kinematic!")

                    target_th1 = np.random.uniform(0, np.deg2rad(10))
                    target_th2 = np.random.uniform(0, np.deg2rad(10))
                    target = rob.get_ee(
                        jangles_rad=[
                            rob.jangles_rad[0] + target_th1,
                            rob.jangles_rad[1] + target_th2,
                        ]
                    )

                    def _cost(x):
                        xy = rob.get_ee(jangles_rad=x)
                        return np.linalg.norm(np.subtract(xy, target))

                    res = minimize(_cost, x0=rob.jangles_rad)
                    target_jangles = res.x
                    print("res", res)

                    rob.plan_move_to(target_jangles, [np.pi, np.pi])

        if target is not None:
            pygame.draw.circle(screen, (255, 0, 0), target, 5)

        rob.update_joints()
        rob.draw(screen)

        pygame.display.update()
        FramePerSec.tick(FPS)


@dataclass
class Movement:
    target_state: t.Any
    speeds: np.ndarray
    last_update: t.Optional[int] = None
    # XXX for both: number of zeros depends on number of motors
    move_elapsed: np.ndarray = field(default_factory=lambda: np.array([0.0, 0]))
    # has to be set the moment the move starts executing, since the
    # robot joint angles can be anything beforehand
    move_planned: np.ndarray = field(default_factory=lambda: np.array([0.0, 0]))


class Robot:
    def __init__(self, xy_origin, lengths):
        self.xy_origin = np.array(xy_origin)
        self.lengths = lengths
        self.jangles_rad = np.zeros(len(self.lengths))
        self.movequeue: t.List[Movement] = []

        self._tips = []

    def plan_move_to(self, angles_rad, speeds_radps):
        assert (
            len(angles_rad) == len(self.lengths) == len(self.jangles_rad)
        ), f"{angles_rad} vs {self.lengths} vs {self.jangles_rad}"

        # single speed will be applied to all joints
        if isinstance(speeds_radps, float):
            speeds_radps = np.array([speeds_radps] * len(angles_rad))

        self.movequeue.append(Movement(angles_rad, speeds_radps))

    def update_joints(self):
        if not self.movequeue:
            return

        active_move = self.movequeue[0]
        now = pygame.time.get_ticks()

        # if we haven't worked in this move yet
        if all(active_move.move_planned == 0):
            planned_move = np.array(active_move.target_state - self.jangles_rad)
            active_move.move_planned = np.abs(planned_move)

            # if speeds are in opposite direction to target, we need to move much further
            where_reverse = planned_move * active_move.speeds < 0
            active_move.move_planned[where_reverse] = (
                2 * np.pi - active_move.move_planned[where_reverse]
            )

        assert len(active_move.move_planned) == len(
            self.lengths
        ), f"planned failed {active_move}"
        assert len(active_move.move_elapsed) == len(
            self.lengths
        ), f"elasped failed {active_move}"

        if all(active_move.move_elapsed >= active_move.move_planned):
            print(
                "popping",
                active_move,
                "had target",
                np.rad2deg(active_move.target_state),
            )
            self.movequeue.pop(0)
            self.jangles_rad = np.array(active_move.target_state)  # cheat
            return

        if active_move.last_update:
            where_needmove = active_move.move_elapsed < active_move.move_planned

            ticks_passed = now - active_move.last_update
            secs_passed = ticks_passed / 1000
            speeds = np.array(active_move.speeds)[where_needmove]
            move_amount = secs_passed * speeds

            self.jangles_rad[where_needmove] += move_amount
            active_move.move_elapsed[where_needmove] += np.abs(move_amount)
            print("jangles", self.jangles_rad)

        active_move.last_update = now

    def draw(self, surface):
        xy = self.xy_origin
        prev_angle = 0

        for joint_angle, joint_len in zip(self.jangles_rad, self.lengths):
            link_vec = np.array(
                [np.cos(joint_angle + prev_angle), np.sin(joint_angle + prev_angle)]
            )
            next_pt = xy + link_vec * joint_len

            armpoly = get_arm_poly(xy, next_pt)
            pygame.draw.aalines(surface, (255, 255, 255), True, armpoly)

            xy = next_pt
            prev_angle = joint_angle

        self._tips.append(xy)
        if len(self._tips) >= 2:
            pygame.draw.aalines(surface, (245, 0, 245, 10), False, self._tips)
            pygame.draw.aalines(surface, (255, 255, 255), False, self._tips[-6:])

        font = pygame.font.SysFont(None, 24)
        img = font.render(
            f"Move Queue Len: {len(self.movequeue)}", True, (255, 255, 255)
        )
        surface.blit(img, (20, 20))

    def get_ee(self, xy_origin=None, lengths=None, jangles_rad=None):
        if xy_origin is None:
            xy_origin = self.xy_origin
        if lengths is None:
            lengths = self.lengths
        if jangles_rad is None:
            jangles_rad = self.jangles_rad

        l1 = lengths[0]
        l2 = lengths[1]

        th1 = jangles_rad[0]
        th2 = jangles_rad[1]

        x0, y0 = xy_origin

        x1 = x0 + l1 * tcos(th1)
        y1 = y0 + l1 * tsin(th1)

        x2 = x1 + l2 * tcos(-(th1 + th2))
        y2 = y1 - l2 * tsin(-(th1 + th2))

        return (x2, y2)


def tsin(a, x=None):
    """Taylor expansion of degree 2 of sin(x) at a"""

    x = x or a
    return math.sin(a) + math.cos(a) * (x - a) + -math.sin(a) / 2 * (x - a) ** 2


def tcos(a, x=None):
    """Taylor expansion of degree 2 of cos(x) at a"""

    x = x or a
    return math.cos(a) - math.sin(a) * (x - a) + -math.cos(a) / 2 * (x - a) ** 2


def get_arm_poly(pt_origin, pt_end):
    """
    Generates the polygon points for visualizing an arm in the Blender
    Armature style
    """

    vec_axial = np.subtract(pt_end, pt_origin).astype(float)
    vec_perp = np.array([vec_axial[1], -vec_axial[0]])
    length = np.linalg.norm(vec_axial)

    vec_axial /= length
    vec_perp /= np.linalg.norm(vec_perp)

    width = 0.2 * length
    stemlen = 0.1 * length

    return np.array(
        [
            pt_origin,
            pt_origin + vec_perp * width / 2 + vec_axial * stemlen,
            pt_end,
            pt_origin - vec_perp * width / 2 + vec_axial * stemlen,
        ]
    ).tolist()


if __name__ == "__main__":
    main()
