import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy

# Head on shooting is bad when balls cant go to any hole
# Make use of side rails bounces


PI = math.pi
ANGLE = numpy.linspace(-1,1,200)
FORCE = numpy.linspace(0,1,100)

def wrap_angle(angle):
    return (angle + PI)%(2*PI) - PI 

def sign(x):
    if x > 0 :
        return 1
    elif x < 0 :
        return -1
    return 0


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
        # for i in range(6):
        #     if abs(PI/2 - abs(theta_BH[i])) < PI/12 :
        #         theta_BH[i] += sign(theta_BH[i])*(PI/2 - abs(theta_BH[i]))*0.8 
        return theta_BH

    def cue_dist(self, ballx, bally, cuex, cuey):
        dist = numpy.square(ballx - cuex) + numpy.square(bally - cuey)
        dist = numpy.sqrt(dist)
        return dist
    
    def get_force(self, distance_to_ball, distance_to_hole = None, valid = True, angle_dev = None):
        if valid :
            # f = (0.3*distance_to_hole**2 + 0.6*distance_to_ball**2)/100000
            f = (0.3*distance_to_hole**2 + 0.65*distance_to_ball**2)/40000
            # f *= angle_dev*10
        else :
            f = distance_to_ball**2/100000
        return f
    
    def pot_ball(self, ballx, bally, cue_angle, d):
        hole_angles = self.hole_angle(ballx, bally)
        theta_m = PI/2 #- math.asin(self.ball_radius * 2/d)
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
            idx = numpy.argmin(numpy.absolute(valid_shot_angles))
            ret_angle = -numpy.min(numpy.absolute(valid_shot_angles)[idx]) 
            # print((-min(numpy.absolute(valid_shot_angles)))*180/PI)
            # ret_angle = -cue_angle -valid_shot_angles[self.closest_hole(ballx, bally, valid_holes)]
            # force = 0.8*d/1000 + 0.1
            # force = 0.5
            d_idx = numpy.sqrt((self.holes[idx,0] - ballx)**2 + (self.holes[idx,1] - bally)**2)
            force = self.get_force(d, distance_to_hole = d_idx, valid = True, angle_dev = ret_angle)
        else :
            # print(" GOING BLIND")
            ret_angle = 0
            # force = 1
            force = self.get_force(d, valid = False)
        ret_angle -= cue_angle
        # print((ret_angle + cue_angle) * 180/PI)
        return [ret_angle, force]
    
    def error_actuation(self, ballx, bally, cue_angle, d):
        hole_angles = self.hole_angle(ballx, bally)
        theta_m =PI/2 - math.asin(self.ball_radius * 2/d)
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
            idx = numpy.argmin(numpy.absolute(valid_shot_angles))
            ret_angle = numpy.min(numpy.absolute(valid_shot_angles)[idx]) 
            # print((-min(numpy.absolute(valid_shot_angles)))*180/PI)
            # ret_angle = -cue_angle -valid_shot_angles[self.closest_hole(ballx, bally, valid_holes)]
            # force = 0.8*d/1000 + 0.1
            # force = 0.5
            d_idx = numpy.sqrt((self.holes[idx,0] - ballx)**2 + (self.holes[idx,1] - bally)**2)
            force = self.get_force(d, distance_to_hole = d_idx, valid = True, angle_dev = ret_angle)
        else :
            # print(" GOING BLIND")
            ret_angle = 0
            # force = 1
            force = self.get_force(d, valid = False)
            ret_angle -= cue_angle
        return [ret_angle, force]
    
    def value(self, ball_pos):
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
        value = 0
        for i, ball in enumerate(balls) :
            angle, force = self.error_actuation(x_coords[i], y_coords[i], theta_CB[i], dist[i])
            if angle != 0 :
                value += 0.0001/abs(angle)
            value -= force/10
        # print(value)
        return value
    
    def choose_ball(self, actions, ball_pos):
        # force = [actions[i][1] for i in actions.keys()]
        # force = numpy.array(force)
        # cost = force + dist/200
        no_of_balls = len(actions.keys())
        curr_value = -float('inf')
        action = 0
        for i in actions.keys():
            next_state = self.ns.get_next_state(ball_pos, actions[i], seed=10)
            value = self.value(next_state)
            if len(next_state.keys())-2 > no_of_balls:
                return actions[i]
            if value > curr_value :
                action = i
                curr_value = value
        # print(curr_value)
        return actions[action]


    def action(self, ball_pos=None):
        self.holes[0,0] = 40 + 24*2
        self.holes[0,1] = 40 + 24*2
        self.holes[1,0] = 40 + 24*2
        self.holes[1,1] = 460 - 24*2
        self.holes[2,0] = 500
        self.holes[2,1] = 40 + 24*2
        self.holes[3,0] = 500
        self.holes[3,1] = 460 - 24*2
        self.holes[4,0] = 960 - 24*2
        self.holes[4,1] = 40 + 24*2
        self.holes[5,0] = 960 - 24*2
        self.holes[5,1] = 460 - 24*2
        print(self.holes)
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
        # chosen_ball = self.choose_ball(actions_for_balls, ball_pos)
        # chosen_ball = 0
        # shot_angle = actions_for_balls[chosen_ball][0]
        # force = actions_for_balls[chosen_ball][1]
        shot_angle, force = self.choose_ball(actions_for_balls, ball_pos)
        # time.sleep(1)
        return (shot_angle/PI ,force)