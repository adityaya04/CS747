import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy
random.seed(32)

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

    def closest_hole(self, ballx, bally, valid_holes):
        hole_dist = self.cue_dist(self.holes[:,0], self.holes[:,1], ballx, bally)
        new_holes_dist = []
        for i in range(6):
            if i in valid_holes:
                new_holes_dist.append(hole_dist[i])
        return numpy.argmin(numpy.array(new_holes_dist))

    def hole_angle(self, ballx, bally, theta_CB = None, d = None):
        hole_x, hole_y = self.holes[:,0], -self.holes[:,1]
        theta_BH = -numpy.arctan2(hole_y + bally, hole_x - ballx) + PI/2
        theta_BH = wrap_angle(theta_BH)
        return theta_BH

    def cue_dist(self, ballx, bally, cuex, cuey):
        dist = numpy.square(ballx - cuex) + numpy.square(bally - cuey)
        dist = numpy.sqrt(dist)
        return dist
    
    def get_force(distance, valid = True):
        if valid :
            if distance > 200 :
                f = 0.5
            else :
                f = 0.25
        else :
            if distance > 200 :
                f = 0.8
            else :
                f = 0.8
    
    def pot_ball(self, ballx, bally, cue_angle, d):
        hole_angles = self.hole_angle(ballx, bally)
        theta_m = PI/2 - math.asin(self.ball_radius * 2/d)
        valid_holes = []
        angle_error = hole_angles - cue_angle
        l = numpy.sqrt(d**2 + 4*(self.ball_radius**2) - 4 * d * self.ball_radius * numpy.cos(angle_error))
        shot_angle = numpy.arcsin(2 * self.ball_radius * numpy.sin(angle_error) / l)
        for i in range(6):
            if abs(angle_error[i]) < theta_m :
                valid_holes.append(i)
        valid_shot_angles = []
        for hole in range(6):
            # if hole not in valid_holes :
                # continue
            valid_shot_angles.append(shot_angle[hole])
            
        ret_angle = 0
        force = 0

        if len(valid_shot_angles) > 0 :
            ret_angle = -min(numpy.absolute(valid_shot_angles)) - cue_angle
            # print((-min(numpy.absolute(valid_shot_angles)))*180/PI)
            # ret_angle = -cue_angle -valid_shot_angles[self.closest_hole(ballx, bally, valid_holes)]
            # force = 0.8*d/1000 + 0.1
            force = 0.5
        else :
            # print(" GOING BLIND")
            ret_angle = -cue_angle
            force = 1

        print((ret_angle + cue_angle) * 180/PI)
        return [ret_angle, force]
    
    def choose_ball(self, actions, dist):
        force = [actions[i][1] for i in actions.keys()]
        force = numpy.array(force)
        cost = force + dist/200
        return numpy.argmin(cost)


    def action(self, ball_pos=None):
        balls = list(ball_pos.keys())
        balls.remove('white')
        if 0 in balls :
            balls.remove(0)
        balls.sort()
        cue_x, cue_y = ball_pos['white']
        x_coords = [ball_pos[i][0] for i in balls]
        y_coords = [ball_pos[i][1] for i in balls]
        dist = self.cue_dist(x_coords, y_coords, cue_x, cue_y)
        theta_CB = -numpy.arctan2(cue_y - y_coords, x_coords - cue_x) + PI/2 
        theta_CB = wrap_angle(theta_CB)
        actions_for_balls = {}
        for i, ball in enumerate(balls) :
            actions_for_balls[i] = self.pot_ball(x_coords[i], y_coords[i], theta_CB[i], dist[i]) # [angle, force]
        chosen_ball = self.choose_ball(actions_for_balls, dist)
        # chosen_ball = 0
        shot_angle = actions_for_balls[chosen_ball][0]
        force = actions_for_balls[chosen_ball][1]
        time.sleep(1)
        return (shot_angle/PI ,force)