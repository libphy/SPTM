# coding: utf-8
import os
import sys
sys.path.append('..')
sys.path.append('../common')
sys.path.append('../train')
from resnet import *
from constants import *
from train_setup import *

action_network = ACTION_NETWORK(((1 + ACTION_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), ACTION_CLASSES)
action_network.summary()
