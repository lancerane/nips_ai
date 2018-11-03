#!/usr/bin/env python

import opensim as osim

from osim.redis.client import Client
from osim.env import *
import numpy as np
import argparse
import os

from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import opensim
import pandas as pd
import numpy as np
from osim.env import ProstheticsEnv
import gym
import tensorflow
from baselines.common.mpi_running_mean_std import RunningMeanStd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype, MultiCategoricalPdType
from osim.http.client import Client

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np, pandas as pd
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import opensim
from easy_tf_log import tflog
from osim.http.client import Client
from osim.env import ProstheticsEnv
from baselines.common.distributions import make_pdtype, MultiCategoricalPdType
from baselines.common.mpi_running_mean_std import RunningMeanStd

import gym

import os

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name


    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)



        self.pdtype = pdtype = make_pdtype(ac_space)
        ### for binary actions #####
        self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                          high=np.ones_like(ac_space.high, dtype=np.int32))
        gaussian_fixed_var= True
        binary = True                       
        #############################                           
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        

        
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0))) #tanh
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0))) #tanh
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box) and binary == False:
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))
                # logstd = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='logstd', kernel_initializer=U.normc_initializer(0.01))
                # pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)



        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, np.expand_dims(ob,0))
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

def ob_dict_to_state(state_desc):

    res=[]

    res += [state_desc["target_vel"][0]- state_desc["body_vel"]['pelvis'][0]]
    res += [state_desc["target_vel"][2]- state_desc["body_vel"]['pelvis'][2]]

    res += [state_desc["target_vel"][0]]
    res += [state_desc["target_vel"][2]]

    pelvis_x_pos = state_desc["body_pos"]["pelvis"][0]
    pelvis_y_pos = state_desc["body_pos"]["pelvis"][1]
    pelvis_z_pos = state_desc["body_pos"]["pelvis"][2]

    for body_part in ["pelvis"]:
        res += state_desc["body_pos_rot"][body_part][:3] #ground_pelvis/pelvis_tilt/value in states file
        res += state_desc["body_vel_rot"][body_part][:3]
        res += state_desc["body_acc_rot"][body_part][:3]#2
        res += state_desc["body_acc"][body_part][0:3]

        #### for cyclical state, need to change pelvis_x_pos
        # res += [pelvis_x_pos]
        #####

        res += [state_desc["body_vel"][body_part][0]]
        res += [pelvis_y_pos]
        res += [state_desc["body_vel"][body_part][1]]
        res += [state_desc["body_vel"][body_part][2]]

    for body_part in ["head","torso", "pros_tibia_r","pros_foot_r","toes_l","talus_l"]:
        res += state_desc["body_pos_rot"][body_part][:3] #ground_pelvis/pelvis_tilt/value in states file
        res += state_desc["body_vel_rot"][body_part][:3]
        res += state_desc["body_acc_rot"][body_part][:3]#2
        res += state_desc["body_acc"][body_part][:3]
        res += [state_desc["body_pos"][body_part][0] - pelvis_x_pos]
        res += [state_desc["body_vel"][body_part][0]]
        res += [state_desc["body_pos"][body_part][1] - pelvis_y_pos]
        res += [state_desc["body_vel"][body_part][1]]
        res += [state_desc["body_pos"][body_part][2] - pelvis_z_pos]
        res += [state_desc["body_vel"][body_part][2]]

    #Only hip has more than one dof, but here last position is locked so not worth including 

    for joint in ["hip_r","knee_r","hip_l","knee_l","ankle_l"]: #removed back
        res += state_desc["joint_pos"][joint][:2]
        res += state_desc["joint_vel"][joint][:2]
        res += state_desc["joint_acc"][joint][:2] 

    mus_list = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l']
    for muscle in mus_list:#state_desc["muscles"].keys():
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        # Add in muscle forces
        # res += state_desc['forces'][muscle]
    res += state_desc["forces"]["ankleSpring"]

    for foot in ['pros_foot_r_0','foot_l']:
        res += state_desc['forces'][foot][:6]

    cm_pos_x = [state_desc["misc"]["mass_center_pos"][0] - pelvis_x_pos]
    cm_pos_y = [state_desc["misc"]["mass_center_pos"][1] - pelvis_y_pos]
    cm_pos_z = [state_desc["misc"]["mass_center_pos"][2] - pelvis_z_pos]
    res = res + cm_pos_x + cm_pos_y + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res


import baselines.common.tf_util as U
rank = MPI.COMM_WORLD.Get_rank()
sess = U.single_threaded_session()
sess.__enter__()

workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
set_global_seeds(workerseed)

g = tf.get_default_graph()
with g.as_default():
    tf.set_random_seed(workerseed)

def policy_fn(name, ob_space, ac_space):
    return crowd_ai_load_and_submit.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=768, num_hid_layers=2)


"""
NOTE: For testing your submission scripts, you first need to ensure 
that redis-server is running in the background
and you can locally run the grading service by running this script : 
https://github.com/crowdAI/osim-rl/blob/master/osim/redis/service.py
The client and the grading service communicate with each other by 
pointing to the same redis server.
"""

"""
Please ensure that `visualize=False`, else there might be unexpected errors 
in your submission
"""
env = ProstheticsEnv(visualize=False)

"""
Define evaluator end point from Environment variables
The grader will pass these env variables when evaluating
"""
REMOTE_HOST = os.getenv("CROWDAI_EVALUATOR_HOST", "127.0.0.1")
REMOTE_PORT = os.getenv("CROWDAI_EVALUATOR_PORT", 6379)
client = Client(
    remote_host=REMOTE_HOST,
    remote_port=REMOTE_PORT
)

# Create environment
observation = client.env_create()

ob_space = env.observation_space
ac_space = env.action_space

g=tf.get_default_graph()
with g.as_default():
    tf.set_random_seed(8)

pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy


atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
# clip_param = clip_param * lrmult # Annealed cliping parameter epislon

ob = U.get_placeholder_cached(name="ob")
ac = pi.pdtype.sample_placeholder([None])

kloldnew = oldpi.pd.kl(pi.pd)
ent = pi.pd.entropy()
meankl = tf.reduce_mean(kloldnew)
meanent = tf.reduce_mean(ent)
# pol_entpen = (-entcoeff) * meanent

ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
surr1 = ratio * atarg # surrogate from conservative policy iteration

U.initialize()

saver=tf.train.Saver()
basePath=os.path.dirname(os.path.abspath(__file__))
modelF= basePath + '/' + "ProstheticsEnv_afterIter_"+str(118) + '.model'
saver.restore(tf.get_default_session(), modelF)
logger.log("Loaded model from {}".format(modelF))

"""
The grader runs N simulations of at most 1000 steps each. 
We stop after the last one
A new simulation starts when `clinet.env_step` returns `done==True`
and all the simulations end when the subsequent `client.env_reset()` 
returns a False
"""
while True:
    res = ob_dict_to_state(observation)
    ac, vpred = pi.act(False, res)

    [observation, reward, done, info] = client.env_step(ac.tolist())
    # print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
