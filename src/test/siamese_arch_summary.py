# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../common')
from resnet import *
from constants import *
siamese  = ResnetBuilder.build_siamese_resnet_18((6,120,160),2)
siamese.summary()
