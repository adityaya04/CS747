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
# Dont choose hole with min angle error

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

    def closest_hole(self, ballx, bally):
        hole_dist = self.cue_dist(self.holes[:,0], self.holes[:,1], ballx, bally)
        return numpy.min((numpy.array(hole_dist)))

    def hole_angle(self, ballx, bally, theta_CB = None, d = None):
        hole_x, hole_y = self.holes[:,0], -self.holes[:,1]
        theta_BH = -numpy.arctan2(hole_y + bally, hole_x - ballx) + PI/2
        theta_BH = wrap_angle(theta_BH)
        hole_fov = PI/6
        correction_factor = 190 * 1e-3
        # if d > 80 :
        #     for i in range(6):
        #         if i == 0 :
        #             if abs(theta_BH[i] + PI/4) > hole_fov/2 :
        #                 # print("correcting hole 0")
        #                 if theta_BH[i] < -PI/4 :
        #                     theta_BH[i] -= abs(theta_BH[i] + PI/4)*correction_factor
        #                 else :
        #                     theta_BH[i] += abs(theta_BH[i] + PI/4)*correction_factor
        #             # theta_BH[i] = min(0,max(theta_BH[i], -PI/2))
        #         elif i == 1 :
        #             if abs(theta_BH[i] + 3*PI/4) > hole_fov/2 :
        #                 # print("correcting hole 1")
        #                 if theta_BH[i] < -3*PI/4 :
        #                     theta_BH[i] -= abs(theta_BH[i] + 3*PI/4)*correction_factor
        #                 else :
        #                     theta_BH[i] += abs(theta_BH[i] + 3*PI/4) *correction_factor
        #             # theta_BH[i] = min(-PI/2,max(theta_BH[i], -PI))
        #         elif i == 4 :
        #             if abs(theta_BH[i] - PI/4) > hole_fov/2 :
        #                 # print("correcting hole 4")
        #                 if theta_BH[i] > PI/4 :
        #                     theta_BH[i] += abs(theta_BH[i] - PI/4)*correction_factor
        #                 else :
        #                     theta_BH[i] -= abs(theta_BH[i] - PI/4)*correction_factor
        #             # theta_BH[i] = min(PI/2,max(theta_BH[i], 0))
        #         elif i == 5 :
        #             if abs(theta_BH[i] - 3*PI/4) > hole_fov/2 :
        #                 # print("correcting hole 5")
        #                 if theta_BH[i] > 3*PI/4 :
        #                     theta_BH[i] += abs(theta_BH[i] - 3*PI/4)*correction_factor
        #                 else :
        #                     theta_BH[i] -= abs(theta_BH[i] - 3*PI/4)*correction_factor
        if d > 150 :         
            dcorner = 1.414
            self.holes[0,0] = 40 + 24*dcorner
            self.holes[0,1] = 40 + 24*dcorner
            self.holes[1,0] = 40 + 24*dcorner
            self.holes[1,1] = 460 - 24*dcorner
            self.holes[2,0] = 500
            self.holes[2,1] = 40 + 24*0
            self.holes[3,0] = 500
            self.holes[3,1] = 460 - 24*0
            self.holes[4,0] = 960 - 24*dcorner
            self.holes[4,1] = 40 + 24*dcorner
            self.holes[5,0] = 960 - 24*dcorner
            self.holes[5,1] = 460 - 24*dcorner
            hole_x, hole_y = self.holes[:,0], -self.holes[:,1]
            theta_BH = -numpy.arctan2(hole_y + bally, hole_x - ballx) + PI/2
            theta_BH = wrap_angle(theta_BH)
        return theta_BH
    
    def hole_dist(self, ballx, bally):
        hole_x, hole_y = self.holes[:,0], self.holes[:,1]
        d = numpy.sqrt((hole_x - ballx)**2 + (hole_y - bally)**2)
        return d

    def cue_dist(self, ballx, bally, cuex, cuey):
        dist = numpy.square(ballx - cuex) + numpy.square(bally - cuey)
        dist = numpy.sqrt(dist)
        return dist
    
    def get_force(self, distance_to_ball, distance_to_hole = None, valid = True, angle_dev = None):
        f = (0.6*distance_to_hole + 0.5*distance_to_ball)/500
        if valid :
            if not distance_to_hole < 50 :
                f /= math.cos(angle_dev)
        return f
    
    def pot_ball(self, ballx, bally, cue_angle, d):
        hole_angles = self.hole_angle(ballx, bally, d=d)
        theta_m = PI/2 - math.asin(self.ball_radius * 2/d)
        valid_holes = []
        angle_error = hole_angles - cue_angle
        hole_dists = self.hole_dist(ballx, bally)
        for i in range(6):
            if abs(angle_error[i]) < theta_m :
                valid_holes.append(i)
            else :
                # valid_holes.append(i)
                angle_error[i] = theta_m * sign(angle_error[i])
        l = numpy.sqrt(d**2 + 4*(self.ball_radius**2) - 4 * d * self.ball_radius * numpy.cos(angle_error))
        shot_angle = numpy.arcsin(2 * self.ball_radius * numpy.sin(angle_error) / l)
            
        valid_shot_angles = []
        valid_hole_dists = []
        for hole in range(6):
            if hole not in valid_holes :
                continue
            valid_shot_angles.append(shot_angle[hole])
            valid_hole_dists.append(hole_dists[hole])
            
        ret_angle = 0
        force = 0
        if len(valid_shot_angles) > 0 :
            cost = numpy.absolute(valid_shot_angles) + 0*numpy.array(valid_hole_dists)/600
            idx = numpy.argmin(cost)
            ret_angle = valid_shot_angles[idx]
            d_idx = numpy.sqrt((self.holes[idx,0] - ballx)**2 + (self.holes[idx,1] - bally)**2)
            force = self.get_force(d, distance_to_hole = d_idx, valid = True, angle_dev = ret_angle)
            # if d_idx < 40 :
            #     ret_angle = 0
        else :
            ret_angle = 0
            d_closest = self.closest_hole(ballx, bally)
            force = self.get_force(d, distance_to_hole=d_closest ,valid = False)
        return [ret_angle, force]
    
    def choose_ball(self, actions, ball_pos):
        force = [actions[i][1] for i in actions.keys()]
        angle = [abs(actions[i][0]) for i in actions.keys()]
        force = numpy.array(force)
        angle = numpy.array(angle)
        alpha = 0.2
        cost = alpha * force + (1 - alpha) * angle
        no_of_balls = len(actions.keys())
        for i in actions.keys():
            next_state = self.ns.get_next_state(ball_pos, [actions[i][0]/PI,actions[i][1]], seed=10)
            if len(next_state.keys())-2 < no_of_balls:
                print("returning from sim")
                return actions[i]
        return actions[numpy.argmin(cost)]
        # return random.choice(actions)


    def action(self, ball_pos=None):
        dcorner = 0
        self.holes[0,0] = 40 + 24*dcorner
        self.holes[0,1] = 40 + 24*dcorner
        self.holes[1,0] = 40 + 24*dcorner
        self.holes[1,1] = 460 - 24*dcorner
        self.holes[2,0] = 500
        self.holes[2,1] = 40 + 24*0
        self.holes[3,0] = 500
        self.holes[3,1] = 460 - 24*0
        self.holes[4,0] = 960 - 24*dcorner
        self.holes[4,1] = 40 + 24*dcorner
        self.holes[5,0] = 960 - 24*dcorner
        self.holes[5,1] = 460 - 24*dcorner
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
            actions_for_balls[i][0] -= theta_CB[i]
            
        # chosen_ball = self.choose_ball(actions_for_balls, ball_pos)
        # chosen_ball = 0
        # shot_angle = actions_for_balls[chosen_ball][0]
        # force = actions_for_balls[chosen_ball][1]
        shot_angle, force = self.choose_ball(actions_for_balls, ball_pos)
        return (shot_angle/PI ,force)