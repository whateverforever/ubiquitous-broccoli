import sys
import typing as t
from dataclasses import dataclass, field

import numpy as np

import pygame

pygame.init()
vec2 = pygame.math.Vector2

WIDTH = 800
HEIGHT = 600
FPS = 60

FramePerSec = pygame.time.Clock()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Arm Plotter")

    rob = Robot([WIDTH // 2, HEIGHT // 2], [200, 100])
    rob.plan_move_to([3.14, 3.14], [2 * np.pi / 10, 2 * np.pi / 5])

    while True:
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

        screen.fill((80, 80, 80))
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

    def plan_move_to(self, angles_rad, speeds_radps):
        assert len(angles_rad) == len(self.lengths) == len(self.jangles_rad)

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
            active_move.move_planned = np.abs(
                active_move.target_state - self.jangles_rad
            )

        assert len(active_move.move_planned) == len(
            self.lengths
        ), f"planned failed {active_move}"
        assert len(active_move.move_elapsed) == len(
            self.lengths
        ), f"elasped failed {active_move}"

        if all(active_move.move_elapsed >= active_move.move_planned):
            print("popping", active_move)
            self.movequeue.pop(0)
            self.jangles_rad = np.array(active_move.target_state)  # cheat
            return

        if active_move.last_update:
            where_needmove = active_move.move_elapsed < active_move.move_planned
            print("where_needmove", where_needmove)

            ticks_passed = now - active_move.last_update
            secs_passed = ticks_passed / 1000
            speeds = np.array(active_move.speeds)[where_needmove]
            move_amount = secs_passed * speeds

            self.jangles_rad[where_needmove] += move_amount
            active_move.move_elapsed[where_needmove] += np.abs(move_amount)

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
            pygame.draw.polygon(surface, (255, 255, 255), armpoly, width=2)

            xy = next_pt
            prev_angle = joint_angle


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
