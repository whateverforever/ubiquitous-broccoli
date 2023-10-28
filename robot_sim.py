import sys

import numpy as np

import pygame
from pygame.locals import *

pygame.init()
vec2 = pygame.math.Vector2

WIDTH = 800
HEIGHT = 600
FPS = 60

FramePerSec = pygame.time.Clock()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Arm Plotter")

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((80, 80, 80))

        # pts = get_arm_poly([100, 100], [170, 230])
        # pygame.draw.polygon(screen, (255, 255, 255), pts, width=1)
        # pts = get_arm_poly([170, 230], [250, 350])
        # pygame.draw.polygon(screen, (255, 255, 255), pts, width=1)

        rob = Robot([200,200], [50, 100, 200])
        rob.draw(screen)

        pygame.display.update()
        FramePerSec.tick(FPS)


class Robot:
    def __init__(self, xy_origin, lengths):
        """
        Arguments
        ---------
        xy_origin:
            Where is the robot base located?
        links:
            List of arms/beams/links lengths
        """

        self.xy_origin = np.array(xy_origin)
        self.lengths = lengths
        self.joint_angles = [0] * len(self.lengths)

    def draw(self, surface):
        xy = self.xy_origin

        for joint_angle, joint_len in zip(self.joint_angles, self.lengths):
            link_vec = np.array([np.cos(joint_angle), np.sin(joint_angle)])
            next_pt = xy + link_vec * joint_len

            armpoly = get_arm_poly(xy, next_pt)
            pygame.draw.polygon(surface, (255, 255, 255), armpoly, width=2)

            xy = next_pt


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

    return np.array([
        pt_origin,
        pt_origin + vec_perp * width / 2 + vec_axial * stemlen,
        pt_end,
        pt_origin - vec_perp * width / 2 + vec_axial * stemlen,
    ])


if __name__ == "__main__":
    main()
