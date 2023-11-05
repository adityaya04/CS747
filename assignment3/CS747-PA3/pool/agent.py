import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy

PI = math.pi

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

    def cue_dist(self, ballx, bally, cuex, cuey):
        dist = numpy.square(ballx - cuex) + numpy.square(bally - cuey)
        dist = numpy.sqrt(dist)
        return dist

    def hole_dists(self, ballx, bally):
        hole_x, hole_y = self.holes[:,0], self.holes[:,1]
        d = numpy.sqrt((hole_x - ballx)**2 + (hole_y - bally)**2)
        return d
    
    def hole_angles(self, ballx, bally):
        hole_x, hole_y = self.holes[:,0], -self.holes[:,1]
        theta_BH = -numpy.arctan2(hole_y + bally, hole_x - ballx) + PI/2
        theta_BH = wrap_angle(theta_BH)
        return theta_BH
    
    def get_delta(self, ballx, bally, hole, d, cue_angle):
        hole_angles = self.hole_angles(ballx, bally)
        hole_angle = hole_angles[hole]
        theta_m = PI/2 - math.asin(self.ball_radius * 2/d)
        angle_error = numpy.clip(hole_angle - cue_angle, -theta_m, theta_m)
        l = numpy.sqrt(d**2 + 4*(self.ball_radius**2) - 4 * d * self.ball_radius * numpy.cos(angle_error))
        shot_angle = numpy.arcsin(2 * self.ball_radius * numpy.sin(angle_error) / l)
        return shot_angle
    
    def simulate(self, ball, delta, current_state):
        forces = numpy.linspace(0.2, 1, 9)
        no_balls = len(current_state.keys())
        curr_min_dist = 2000
        optimal_force = 0.5
        for force in forces :
            next_state = self.ns.get_next_state(current_state, [-delta,force], seed=10)
            if len(next_state.keys()) < no_balls :
                return force, 0
            ballx = next_state[ball][0]
            bally = next_state[ball][1]
            hole_dists = self.hole_dists(ballx, bally)
            min_dist = numpy.min(hole_dists)
            if min_dist < curr_min_dist :
                optimal_force = force
                curr_min_dist = min_dist
        return optimal_force, min_dist
            

    def action(self, ball_pos):
        balls = list(ball_pos.keys())
        balls.remove('white')
        if 0 in balls :
            balls.remove(0)
        balls.sort()
        cue_x, cue_y = ball_pos['white']
        X = [ball_pos[i][0] for i in balls]
        Y = [ball_pos[i][1] for i in balls]
        dist = self.cue_dist(X, Y, cue_x, cue_y)
        theta_CB = -numpy.arctan2(cue_y - Y, X - cue_x) + PI/2 
        theta_CB = wrap_angle(theta_CB)
        shooting_angles = []
        forces = []
        min_dists = []
        for i, ball in enumerate(balls) :
            alpha = 0.7 # 0.7 best by far
            cost = alpha*self.hole_dists(X[i], Y[i])/600 + (1-alpha)*numpy.absolute(self.hole_angles(X[i], Y[i])-theta_CB[i])
            hole = numpy.argmin(cost)
            # print(f"Hole {hole} for Ball {ball}")
            delta = self.get_delta(X[i], Y[i], hole, dist[i], theta_CB[i])
            delta = theta_CB[i] - delta
            force, closest_dist = self.simulate(ball, delta/PI, ball_pos)
            shooting_angles.append(delta)
            forces.append(force)
            min_dists.append(closest_dist)
        action = numpy.argmin(numpy.array(min_dists))
        return -shooting_angles[action]/PI, forces[action]