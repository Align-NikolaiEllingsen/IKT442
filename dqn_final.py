#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import threading
from collections import deque
import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import tensorflow as tf
import time
import base64
import socket
import sys
import csv
import keyboard

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hides useless log print

filename = "reward_log.csv"

sendtlm = []

# Set udp communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_address = ('127.0.0.1', 5000)
send_address = ('127.0.0.1', 5001 )
sock.bind(receive_address)

time_dqn = time.time()
time_skt = time.time()

ACTIONS = 4

GAMMA = 0.99  
OBSERVE = 2000
EXPLORE = 50000
FINAL_EPSILON = 0.001 
INITIAL_EPSILON = 0.1  
REPLAY_MEMORY = 5000  
BATCH = 32
FRAME_PER_ACTION = 1

throttle = 1
brake = 0
steering_angle = 0
handbrake = 0

data_collect = []

x_t = cv2.imread("lineimage.jpg")
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
s_t1=s_t
r_t = 0

throttle_tlm = 0
brake_tlm = 0
steering_tlm = 0
handbrake_tlm = 0
reward_tlm = 0
camera_tlm = 0
reward_total = 0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4)+ b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def log_results(filename, data_collect):
    # Save the results to a file so we can graph it later.
    with open('C:/Users/Nikolai Ellingsen/IKT442 - Project Deep Q/logs/reward_log.csv', 'w', newline='') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)

    D = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    global time_dqn
    global s_t
    while True:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        global throttle
        global brake
        global steering_angle

        if a_t[0]==1:
            throttle += 0.1
            brake = 0

        if a_t[1]==1:
            brake += 0.1
            throttle = 0

        if a_t[2]==1:
            steering_angle += 0.1

        if a_t[3]==1:
            steering_angle -= 0.1

        

        global time_dqn
        global r_t
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


        while (time_dqn>=time_skt):
            i = 1

        time_dqn=time.time()       

        # store the transition in D
        r_t = reward_tlm

        D.append((s_t, a_t, r_t, s_t1, 0))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        i=0
        while(i<10000):
            i+=1

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
        time_dqn = time.time()

        # update
        s_t = s_t1
        t += 1

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))

        data_collect.append([t, reward_total])
        log_results(filename, data_collect)


def telemetry():
    while True:
        data, address = sock.recvfrom(60000) # Size of allowed packet, set high just in case
        data = data.decode().split(";")

        if data:
            global time_skt
            time_skt=time.time()

            throttle_tlm = data[1]
            brake_tlm = data[2]
            steering_tlm = data[3]
            handbrake_tlm = data[4]

            global reward_tlm
            reward_tlm = float(data[5])

            global reward_total
            reward_total = float(data[6])

            global x_t
            x_t = cv2.imdecode(np.frombuffer(base64.b64decode(data[8]), dtype=np.uint8), flags=cv2.IMREAD_COLOR)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        
            #ret, x_t = cv2.threshold(x_t, 170, 255, cv2.THRESH_BINARY)
      
            global s_t1
            x_t = np.reshape(x_t, (80, 80, 1)) 
            s_t1 = np.append(x_t, s_t[:, :, :3], axis=2)

        else:
            print("hmmmm")

        global throttle,brake,steering_angle,handbrake
        send_control(throttle,brake,steering_angle,handbrake)

def send_control(throttle,brake,steering_angle,handbrake):
    global i
    msg = str(throttle) + ";" + str(brake) + ";" + str(steering_angle) + ";" + str(handbrake)
    sent = sock.sendto(msg.encode(), send_address)

def loop():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

t = threading.Thread(target=loop, name='TrainingLoop')
t2 = threading.Thread(target=telemetry, name='TelemetryLoop')

t.start()
t2.start()

t.join()
t2.join()