import torch
import torch.nn as nn
import sparseconvnet as scn
from layers.conv_GRU import ConvGRU


class SparseRNNObjectDet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True, freeze_layers=False):
        super(SparseRNNObjectDet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        self.sparse_to_dense = scn.sparseToDense.SparseToDense(2, 256)
        self.dense_to_sparse = scn.denseToSparse.DenseToSparse(2)
        dimension = 2
        self.num_recurrent_units = 1

        sparse_out_channels = 256
        self.sparseModel = scn.SparseVggNet(2, nInputPlanes=nr_input_channels, layers=[
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 128], ['C', 128], 'MP',
            ['C', 256], ['C', 256]]
            ).add(scn.Convolution(2, 256, sparse_out_channels, 3, filter_stride=2, bias=False)
            ).add(scn.BatchNormReLU(sparse_out_channels)
            )

        if small_out_map:
            self.cnn_spatial_output_size = [5, 7]
        else:
            self.cnn_spatial_output_size = [6, 8]

        # Layer 8 - Gated Recurrent Unit
        kernel_size = 3
        self.GRU1 = ConvGRU(dimension, sparse_out_channels, sparse_out_channels, kernel_size)

        # Layer 9 - Gated Recurrent Unit
        #kernel_size = 3
        #self.GRU2 = ConvGRU(dimension, nPlanes, out_channels, kernel_size)


        # Layer 10 - output layers
        self.sparsetodense = scn.SparseToDense(2, sparse_out_channels)
        self.cnn_spatial_output_size = [5, 7]
        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = spatial_size_product * 256
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))
        
        #if freeze_layers == True:
            # Load convolutional layers of model
        #    pth = 'log/RNN_sequential_combineddataset_98val_15test/checkpoints/model_step_17.pth'
        #    self.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])
            
            # Freeze convolutional and linear layers
        #    for child in self.sparseModel.children():
        #        for param in child.parameters():
        #            param.requires_grad = False
            #for child in self.linear_1.children():
            #    for param in child.parameters():
            #        param.requires_grad = False
            #for child in self.linear_2.children():
            #    for param in child.parameters():
            #        param.requires_grad = False

    def forward(self, x, prev_states=None):

        if prev_states is None:
            prev_states = None

        states = []
        state_idx = 0

        x = self.inputLayer(x)
        x = self.sparseModel(x)

        x = self.GRU1(x, prev_states)
        state_idx += 1
        states = x

        #x = self.GRU2(x, prev_states[state_idx])
        #state_idx += 1
        #states.append(x)

        x = self.sparsetodense(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

        return x, states
    
    def reset_states(self, mask, prev_states):
        
        if prev_states is None:
            dense_states = torch.zeros([15, 256, 5, 7], device='cuda:0')
        else:
            dense_states = self.sparse_to_dense(prev_states)
            for i, flag in enumerate(mask):
                if flag == 1:
                    dense_states[i] = torch.zeros([256, 5, 7], device='cuda:0')
                
        sparse_states = self.dense_to_sparse(dense_states)
        return sparse_states
        
    
######## MANUAL IMPLEMENTATION #############

        '''
        sparse_out_channels = 256

        nPlanes = nr_input_channels

        # Layer 1 - submanifold convolution
        self.conv1 = scn.Sequential()
        out_channels = 16
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 16
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 2 - submanifold convolution
        out_channels = 32
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 32
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 3 - submanifold convolution
        out_channels = 64
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 64
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 4 - submanifold convolution
        out_channels = 128
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 128
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        self.conv1.add(scn.MaxPooling(dimension, 3, 2))
        nPlanes = out_channels
        
        # Layer 5 - submanifold convolution
        out_channels = 256
        kernel_size = 3
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels
        self.conv1.add(scn.BatchNormReLU(nPlanes))
        out_channels = 256
        self.conv1.add(scn.SubmanifoldConvolution(dimension, nPlanes, out_channels, kernel_size, False))
        nPlanes = out_channels

        # Layer 7 - Strided convolution
        out_channels = 256
        kernel_size = 3
        self.conv1.add(scn.Convolution(dimension, nPlanes, out_channels, kernel_size, filter_stride=2, bias=False))
        self.conv1.add(scn.BatchNormReLU(out_channels))
        '''