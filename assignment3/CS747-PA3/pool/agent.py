import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy
random.seed(73)

PI = math.pi
ANGLE = numpy.linspace(-1,1,200)
FORCE = numpy.linspace(0,1,100)

def wrap_angle(angle):
    return (angle + PI)%(2*PI) - PI 


class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.prev_balls = None
        self.ns = utils.NextState()


    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.holes = numpy.array(self.holes)
        self.ball_radius = radius

    def closest_hole(self, ballx, bally):
        hole_dist = self.cue_dist(self.holes[:,0], self.holes[:,1], ballx, bally)
        return numpy.argmin(hole_dist)

    def hole_angle(self, ballx, bally, theta_CB, d):
        hole_x, hole_y = self.holes[:,0], -self.holes[:,1]
        theta_BH = -numpy.arctan2(hole_y + bally, hole_x - ballx) + PI/2
        theta_BH = wrap_angle(theta_BH)
        theta_e = theta_BH - theta_CB
        theta_e = wrap_angle(theta_e)
        theta_m = PI/2 - math.asin(self.ball_radius * 2/d)
        eligible_indices = numpy.where(abs(theta_e) < theta_m)
        l = numpy.sqrt(d**2 + 4*(self.ball_radius**2) - 4 * d * self.ball_radius * numpy.cos(theta_e))
        theta_C = numpy.arcsin(2 * self.ball_radius * numpy.sin(theta_e) / l)
        return theta_C, eligible_indices


    def cue_dist(self, ballx, bally, cuex, cuey):
        dist = numpy.square(ballx - cuex) + numpy.square(bally - cuey)
        dist = numpy.sqrt(dist)
        return dist
    
    def pot_ball(self, ballx, bally, cuex, cuey):



    def get_shot_angles(self, ball_pos):
        balls = list(ball_pos.keys())
        balls.remove('white')
        if 0 in balls :
            balls.remove(0)
        cue_x, cue_y = ball_pos['white']
        balls.sort()
        x_coords = [ball_pos[i][0] for i in balls]
        y_coords = [ball_pos[i][1] for i in balls]
        dist = self.cue_dist(x_coords, y_coords, cue_x, cue_y)
        shot_dict = {}
        theta_CB = -numpy.arctan2(cue_y - y_coords, x_coords - cue_x) + PI/2 
        theta_CB = wrap_angle(theta_CB)
        for i, ball in enumerate(balls): # Might have to do balls enum because ball index and ball is not same            
            shot_angles, indices = self.hole_angle(x_coords[i], y_coords[i], theta_CB[i], dist[i])
            shot_dict[ball] = shot_angles
        self.prev_balls = len(balls)
        return shot_dict, theta_CB


    def action(self, ball_pos=None):
        balls = list(ball_pos.keys())
        balls.remove('white')
        if 0 in balls :
            balls.remove(0)
        cue_x, cue_y = ball_pos['white']
        balls.sort()
        x_coords = [ball_pos[i][0] for i in balls]
        y_coords = [ball_pos[i][1] for i in balls]




        
        shot_dict, CB_angle = self.get_shot_angles(ball_pos)
        target = list(shot_dict.keys())[0]
        shot_angle = shot_dict[target][0] + CB_angle[0]
        force = 0.5
        # print(shot_dict)
        return (shot_angle/PI ,force)
