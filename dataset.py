#!/usr/bin/env python
# coding: utf-8

# # Dataset Generator
#
# **Abhinav Madahar (<abhinav.madahar@rutgers.edu>) &middot; Sungjin Ahn**
#
# This just generates the dataset used in my Reproduction notebook.

import itertools
import math
import random
from collections import namedtuple
from enum import Enum
from math import ceil, sqrt
from sys import argv

import numpy as np

random.seed(0)

def flatten(nested):
    values = []
    for element in nested:
        try:
            values = values.concat(flatten(element))
        except:
            values.append(element)
    return values

def sgn(val):
    if val > 0:
        return 1
    elif val == 0:
        return 0
    else:
        return -1

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        return Vector2D(self.x + v.x, self.y + v.y)

    def __mul__(self, a):
        return Vector2D(a * self.x, a * self.y)

    def __repr__(self):
        return '{}({}, {})'.format(type(self).__name__, self.x, self.y)

    def __eq__(self, v):
        return self.x == v.x and self.y == v.y

class Position(Vector2D):
    pass

class Velocity(Vector2D):
    pass

def random_position(height, width):
    """Generates a psuedorandom position where 0 <= x < height and 0 <= y < width. The x and y are integers."""
    return Position(random.randrange(height), random.randrange(width))

def random_velocity(speed):
    theta = random.uniform(-1, 1) * math.pi
    return Velocity(speed * math.cos(theta), speed * math.sin(theta))

class GameState(Enum):
    ONGOING = 0
    WON = 1
    COLLIDED = 2
    TIMEOUT = 3

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

ObstacleClass = namedtuple('ObstacleClass', ['length', 'speed'])
Obstacle = namedtuple('Obstacle', ['obstacle_class', 'position'])

def obstacle_from_class_id(id):
    """The `id` argument is the class ID in [0, 1, 2, 3, 4]."""
    obstacle_classes = [
        ObstacleClass(3, 1),
        ObstacleClass(3, 2),
        ObstacleClass(6, 1),
        ObstacleClass(6, 2),
        ObstacleClass(1.5, 0.5),
    ]
    return obstacle_classes[id]

def random_obstacle_lanes(n_lanes):
    return [obstacle_from_class_id(random.randrange(5)) for _ in range(n_lanes)]


class Environment:
    def __init__(self, height, width, agent_pos, goal_pos, agent_speed, goal_vel, max_time, level):
        """
            Initialize the environment.

            Args:
                height: int, the height of the environment.
                width: int, the width of the environment.
                agent_pos: Position, the initial position of the agent.
                goal_pos: Position, the initial position of the goal; the goal is 2x2, so the goal_pos is its bottom-left corner.
                agent_speed: float, the speed (not velocity) of the agent. The model controls the direction.
                goal_vel: Velocity, the velocity of the goal at the start of the experiment. This can change if it hits a wall.
                max_time: int, the maximum number of timesteps which can elapse before an automatic loss.
        """

        self.height = height
        self.width = width

        assert 0 <= agent_pos.x <= width
        assert 0 <= agent_pos.y <= height
        assert 0 <= goal_pos.x <= width - 1
        assert 0 <= goal_pos.y <= height - 1  # ditto
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos

        self.agent_speed = agent_speed
        self.goal_vel = goal_vel

        self.max_time = max_time
        self.time = 0

        # randomly assign obstacle lanes for each lane.
        # note that a single lane can have multiple obstacles, so we have a list of obstacles in each lane
        n_obstacle_lanes = self.height
        self.n_obstacle_lanes = n_obstacle_lanes
        self.obstacle_classes = random_obstacle_lanes(n_obstacle_lanes)
        self.obstacle_directions = [Direction(random.randrange(2)) for _ in range(n_obstacle_lanes)]
        self.obstacle_lanes = [[] for _ in range(n_obstacle_lanes)]

        self.level = level
        self.add_obstacles()

    def __repr__(self):
        return 'Environment(height={}, width={}, agent_pos={}, goal_pos={}, agent_speed={}, goal_vel={})'.format(
            self.height, self.width, self.agent_pos, self.goal_pos, self.agent_speed, self.goal_vel)

    def __str__(self):
        """An ASCII diagram of the environment."""

        board = '|' + '-' * self.width + '|\n'
        for row in range(self.height):
            board += '|'
            for col in range(self.width):
                if Position(row, col) == self.rounded(self.agent_pos):
                    board += 'a'
                elif Position(row, col) in self.hitbox(self.goal_pos, (2, 2)):
                    board += 'g'
                else:
                    board += ' '
            board += '|\n'
        board += '|' + '-' * self.width + '|'

        return board

    def move(self):
        """Makes all the objects move in a single timestep and returns the game state."""

        def move_object(obj, vel):
            # instead of making a good solution, I made an easy one.
            # we move the object by velocity * dt where dt is a small scalar.
            # if we go out of bounds, then we flip the coordinate(s) which are out of bounds
            dt = 0.01
            for _ in range(int(1 / dt)):
                obj += vel * dt

                if obj.x < 0:
                    vel.x = abs(vel.x) * 1  # make sure to go right
                elif obj.x > self.width:
                    vel.x = abs(vel.x) * -1  # make sure to go left

                if obj.y < 0:
                    vel.y = abs(vel.y) * 1  # make sure to go up
                elif obj.y > self.height:
                    vel.y = abs(vel.y) * -1  # make sure to go down


            return obj

        # we move the agent and goal
        self.agent_pos = move_object(self.agent_pos, self.model_decision())
        self.goal_pos = move_object(self.goal_pos, self.goal_vel)

        # we move the obstacles
        for lane, direction in zip(self.obstacle_lanes, self.obstacle_directions):
            for j, obstacle in enumerate(lane):
                speed     = obstacle.obstacle_class.speed
                direction = 1 if direction == Direction.RIGHT else -1
                obstacle.position.y += direction * speed

        self.remove_out_of_bounds_obstacles()
        self.add_obstacles()

        self.time += 1

        if self.rounded(self.agent_pos) in self.hitbox(self.goal_pos, [2, 2]):
            return GameState.WON

        if self.time >= self.max_time:
            return GameState.TIMEOUT

        return GameState.ONGOING

    def hitbox(self, position, shape: [int, int]):
        """
        Returns an array of all the positions which are in this element's hitbox.
        Note that shape = [length in x, height in y].
        """

        position = self.rounded(position)
        hitboxes = []
        for offset_y in range(ceil(shape[0])):
            for offset_x in range(ceil(shape[1])):
                hitboxes.append(Position(position.x + offset_x, position.y + offset_y))
        return hitboxes

    def rounded(self, position):
        return Position(round(position.x), round(position.y))

    def model_decision(self):
        """Get the model's decision on where to move given the current environment using integers 0 through 7."""
        # until I replicate the model, it will always select action 0
        decision = 0
        decisions = [
            Velocity(-1/sqrt(2), 1/sqrt(2)),  Velocity(0, 1),  Velocity(-1/sqrt(2), -1/sqrt(2)),
            Velocity(-1, 0),                                   Velocity(0, 1),
            Velocity(-1/sqrt(2), -1/sqrt(2)), Velocity(0, -1), Velocity(-1/sqrt(2), -1/sqrt(2)),
        ]
        return decisions[decision] * self.agent_speed

    def add_obstacles(self):
        """Add obstacles based on a Poisson distribution."""

        # decide into which lanes to add obstacles
        number_obstacles_to_introduce = np.random.poisson(lam=self.level)
        obstacle_lanes = list(range(self.n_obstacle_lanes))
        random.shuffle(obstacle_lanes)
        lanes_in_which_we_add_obstacles = obstacle_lanes[:number_obstacles_to_introduce]

        for lane in lanes_in_which_we_add_obstacles:
            end = 0 if self.obstacle_directions[lane] == Direction.RIGHT else self.width
            obstacle = Obstacle(self.obstacle_classes[lane], Position(lane, end))
            self.obstacle_lanes[lane].append(obstacle)

    def remove_out_of_bounds_obstacles(self):
        for lane, direction in zip(self.obstacle_lanes, self.obstacle_directions):
            for j, obstacle in enumerate(lane):
                left_moving_and_out_of_bounds = direction == Direction.LEFT and obstacle.position.x + obstacle.obstacle_class.length < 0
                right_moving_and_out_of_bounds = direction == Direction.RIGHT and obstacle.position.x + obstacle.obstacle_class.length > self.width

                if left_moving_and_out_of_bounds or right_moving_and_out_of_bounds:
                    lane.pop(j)

    def model_readable_representation(self):
        """Represents the environment using a numpy tensor. A point is 0 if empty, 1 if an obstacle is there, and 2 for goal."""

        # we have to increment because (self.width, self.height) is a valid point on the board
        rep = np.zeros([self.width+1, self.height+1], dtype=np.float64)

        obstacles = []
        for lane in self.obstacle_lanes:
            for obstacle in lane:
                obstacles.append(obstacle)
        obstacle_points = []
        for obstacle in obstacles:
            for point in environment.hitbox(obstacle.position, [obstacle.obstacle_class.length, 1]):
                obstacle_points.append(point)

        for position in obstacle_points:
            # obstacles can sometimes go beyond the bounds along the x-axis,
            # which is valid but makes it impossible to put them on the representation.
            if 0 <= position.x <= self.width and 0 <= position.y <= self.height:
                rep[position.x][position.y] = 0.5

        goal_hitpoints = self.hitbox(self.goal_pos, [2, 1])
        for position in goal_hitpoints:
            if 0 <= position.x < self.width and 0 <= position.y <= self.height:
                rep[position.x][position.y] = 1

        return rep


height = 45
width = 45
max_time = 407
goal_speed = 1
agent_pos = random_position(height, width)
goal_pos = random_position(height-1, width-1)
agent_speed = 1
goal_vel = random_velocity(goal_speed)
level = 1


n_data_points = int(argv[1])

try:
    X = np.load('autoencoder-data.npy')
    if len(X) > n_data_points:
        X = X[:n_data_points]
    elif len(X) < n_data_points:
        # we have to make more
        raise ValueError('insufficiently-many frames in dataset.')
except (FileNotFoundError, ValueError) as e:
    if isinstance(e, FileNotFoundError):
        # if we couldn't load the file, then we need to make an entire dataset from scratch
        X = np.zeros((n_data_points, (height + 1) * (width + 1)), dtype=np.float32)
        n_data_points_collected = 0
    else:
        # we extend the old X to have enough space for the new dataset
        old_X = X
        X = np.zeros((n_data_points, (height + 1) * (width + 1)), dtype=np.float32)
        for i, frame in enumerate(old_X):
            X[i] = frame
        n_data_points_collected = len(old_X)

    while n_data_points_collected < n_data_points:
        environment = Environment(height, width, agent_pos, goal_pos, agent_speed, goal_vel, max_time, level)
        rep = environment.model_readable_representation().flatten()
        X[n_data_points_collected] = rep
        n_data_points_collected += 1
        while n_data_points_collected < n_data_points:
            X[n_data_points_collected] = environment.model_readable_representation().flatten()
            environment.move()
            n_data_points_collected += 1
    np.save('autoencoder-data.npy', X)
