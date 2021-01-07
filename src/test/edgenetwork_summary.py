# coding: utf-8
import os
import sys
sys.path.append('..')
sys.path.append('../common')
from resnet import *
edge_model = ResnetBuilder.build_siamese_resnet_18((6,120,160), 2)
edge_model.summary()
