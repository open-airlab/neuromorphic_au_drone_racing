import torch
import torch.nn as nn
import torch.nn.functional as f 
from torch.nn import init

import sparseconvnet as scn
from sparseconvnet.activations import Sigmoid, Tanh

import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
        return tempTimeInterval

def tic():
    # Records a time in TicToc, marks thex_asyn[1].unsqueeze(0) beginning of a time interval
    toc(False)


class ConvGRU(nn.Module):
    """
    Generate a sparse convolutional GRU cell. 
    Adapted from https://github.com/cedric-scheerlinck/rpg_e2vid/blob/cedric/firenet/model/submodules.py
    """

    def __init__(self, dimension, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparse_to_dense = scn.sparseToDense.SparseToDense(2, input_size)
        self.dense_to_sparse = scn.denseToSparse.DenseToSparse(2)
        self.reset_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, True)
        self.update_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, True)
        self.out_gate = scn.SubmanifoldConvolution(dimension, input_size + hidden_size, hidden_size, kernel_size, True)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
            
    def input_spatial_size(self, out_size):
        return out_size

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.batch_size()
        spatial_size = input_.spatial_size
        dense_input = self.sparse_to_dense(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            none_prev = True
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            # if torch.cuda.is_available():
            #     dense_prev = torch.zeros(state_size, device='cuda:0')
            # else:
            #     dense_prev = torch.zeros(state_size, device='cpu')
            
            dense_prev = torch.zeros(state_size, device='cpu')
            
            prev_state = self.dense_to_sparse(dense_prev)
        else:
            none_prev = False
            dense_prev = self.sparse_to_dense(prev_state)
            
        stacked_dense_inputs = torch.cat([dense_input, dense_prev], dim=1)
        stacked_inputs = self.dense_to_sparse(stacked_dense_inputs)

        update = self.sigmoid(self.update_gate(stacked_inputs))
        reset = self.sigmoid(self.reset_gate(stacked_inputs))

        if none_prev:
            out_inputs = self.tanh(self.out_gate(stacked_inputs))
        else:
            dense_reset = self.sparse_to_dense(reset)
            prev_reset_mul = dense_prev * dense_reset
            stacked_dense_intermediate_inputs = torch.cat([dense_input, prev_reset_mul], dim=1)
            stacked_intermediate_inputs = self.dense_to_sparse(stacked_dense_intermediate_inputs)
            out_inputs = self.tanh(self.out_gate(stacked_intermediate_inputs))

        if none_prev:       
            new_state = multiply_feature_planes((out_inputs, update))
        else:
            dense_update = (1 - self.sparse_to_dense(update))
            new_state_1_dense = dense_prev * dense_update
            #new_state_1 = self.dense_to_sparse(new_state_1_dense)
            
            new_state_2 = multiply_feature_planes((out_inputs, update))
            new_state_2_dense = self.sparse_to_dense(new_state_2)
            
            new_state_dense = new_state_1_dense + new_state_2_dense
            new_state = self.dense_to_sparse(new_state_dense)
            #new_state = scn.add_feature_planes((new_state_1, new_state_2))

        return new_state


def multiply_feature_planes(input):
    output = scn.SparseConvNetTensor()
    output.metadata = input[0].metadata
    output.spatial_size = input[0].spatial_size
    output.features = input[0].features * input[1].features
    return output


def subtract_constant_feature_planes(input, a):
    output = scn.SparseConvNetTensor()
    output.metadata = input.metadata
    output.spatial_size = input.spatial_size
    output.features = a - input.features
    return output

def concatenate_feature_planes(x, y):
    output = SparseConvNetTensor()
    
    cL,cR,L,R = x.metadata.compareSparseHelper(y.metadata, x.spatial_size)
    output.metadata = x.metadata
    for r in R:
        x.metadata.appendMetadata(y.metadata, x.spatial_size)
    ## merge metadata her
    output.spatial_size = input[0].spatial_size
    output.features = torch.cat([i.features for i in input], 1)
    return output

def concatenate_feature_planes_easy(x, y):
    output = SparseConvNetTensor()
    output.metadata = x.metadata
    x.metadata.appendMetadata(y.metadata, x.spatial_size)
    ## merge metadata her
    output.spatial_size = x.spatial_size
    output.features = torch.cat([i.features for i in input], 1)
    return output
    

