# coding: utf-8
import os
import sys
sys.path.append('..')
sys.path.append('../common')
from sptm import *
from resnet import *
from constants import *
from test_navigation_quantitative import *
os.environ['CUDA_VISIBLE_DEVICES']="1"
environment = 'deepmind_small'
keyframes, keyframe_coordinates, _ = main_exploration(None, environment)
len(keyframes)
keyframes[0].shape
memory = SPTM()
memory.set_shortcuts_cache_file(environment) 
memory.set_memory_buffer(keyframes)
shortcuts_matrix = [] 
#in .self.predict_single_input(self, input)
input_ = keyframes[0]
input_ = memory.input_processor.preprocess_input(input_) #changed variable name input to input_
print('input_', input_.shape)
input_code = np.squeeze(memory.input_processor.bottom_network.predict(np.expand_dims(input_, axis=0), batch_size=1))
print('input_code', input_code.shape)
intermed = memory.input_processor.bottom_network.predict(np.expand_dims(input_, axis=0), batch_size=1)
print('bottom_net out', intermed.shape)
print('tensor_to_predict', memory.input_processor.tensor_to_predict.shape)
memory.input_processor.tensor_to_predict[0][0:input_code.shape[0]]=input_code
memory.input_processor.top_network.summary()
# the last line in the predict_single_input() is broken- top_network from resnet build is dense layers whereas the output of the bottom_network is the first conv output of the edge network. The shape mismatch, and the entire setting doesn't make sense. maybe the definition (though it's from the git) of top and bottom networks are wrong.
