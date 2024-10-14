import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.BasicBlock import MyInception_1, Pyramid_1
import numpy as np
import pynvml
pynvml.nvmlInit()
gpu_index = 0

def print_gpu_memory(device=None):
    if device is None:
        device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    print(f"GPU {device}:")
    print(f"  Total Memory: {mem_info.total / (1024 ** 2):.2f} MB")
    print(f"  Used Memory: {mem_info.used / (1024 ** 2):.2f} MB")
    print(f"  Free Memory: {mem_info.free / (1024 ** 2):.2f} MB")
    print(f"  GPU Utilization: {utilization.gpu}%")
    print(f"  Memory Utilization: {utilization.memory}%")
    print(f"  Temperature: {temperature} C")
    print("-" * 30)

# class MyNet(ME.MinkowskiNetwork):
#     CHANNELS = [None, 32, 32, 64, 128, 256]
#     TR_CHANNELS = [None, 32, 32, 64, 128, 256]
#     BLOCK_1 = MyInception_1
#     BLOCK_2 = Pyramid_1
#
#     def __init__(self,
#                  in_channels=1,
#                  out_channels=1,
#                  bn_momentum=0.1,
#                  last_kernel_size=5,
#                  D=3):
#
#       ME.MinkowskiNetwork.__init__(self, D)
#       CHANNELS = self.CHANNELS
#       TR_CHANNELS = self.TR_CHANNELS
#       BLOCK_1 = self.BLOCK_1
#       BLOCK_2 = self.BLOCK_2
#
#       self.conv1 = ME.MinkowskiConvolution(
#           in_channels=3,
#           out_channels=CHANNELS[1],
#           kernel_size=5,
#           stride=1,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)
#       self.block1 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[1], bn_momentum=bn_momentum, D=D)
#
#       self.conv2 = ME.MinkowskiConvolution(
#           in_channels=CHANNELS[1],
#           out_channels=CHANNELS[2],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
#       self.block2 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[2], bn_momentum=bn_momentum, D=D)
#
#       self.conv3 = ME.MinkowskiConvolution(
#           in_channels=CHANNELS[2],
#           out_channels=CHANNELS[3],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)
#       self.block3 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[3], bn_momentum=bn_momentum, D=D)
#
#       self.conv4 = ME.MinkowskiConvolution(
#           in_channels=CHANNELS[3],
#           out_channels=CHANNELS[4],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm4 = ME.MinkowskiBatchNorm(CHANNELS[4], momentum=bn_momentum)
#       self.block4 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[4], bn_momentum=bn_momentum, D=D)
#
#       self.conv5 = ME.MinkowskiConvolution(
#           in_channels=CHANNELS[4],
#           out_channels=CHANNELS[5],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm5 = ME.MinkowskiBatchNorm(CHANNELS[5], momentum=bn_momentum)
#       self.block5 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[5], bn_momentum=bn_momentum, D=D)
#
#       self.conv5_tr = ME.MinkowskiConvolutionTranspose(
#           in_channels=CHANNELS[5],
#           out_channels=TR_CHANNELS[5],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm5_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[5], momentum=bn_momentum)
#       self.block5_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)
#
#       self.conv4_tr = ME.MinkowskiConvolutionTranspose(
#           in_channels=CHANNELS[4] + TR_CHANNELS[5],
#           out_channels=TR_CHANNELS[4],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm4_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[4], momentum=bn_momentum)
#       self.block4_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)
#
#       self.conv3_tr = ME.MinkowskiConvolutionTranspose(
#           in_channels=CHANNELS[3] + TR_CHANNELS[4],
#           out_channels=TR_CHANNELS[3],
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm3_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[3], momentum=bn_momentum)
#       self.block3_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)
#
#       self.conv2_tr = ME.MinkowskiConvolutionTranspose(
#           in_channels=CHANNELS[2] + TR_CHANNELS[3],
#           out_channels=TR_CHANNELS[2],
#           kernel_size=last_kernel_size,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[2], momentum=bn_momentum)
#       self.block2_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)
#
#       self.conv1_tr = ME.MinkowskiConvolution(
#           in_channels=TR_CHANNELS[2],
#           out_channels=TR_CHANNELS[1],
#           kernel_size=3,
#           stride=1,
#           dilation=1,
#           bias=False,
#           dimension=D)
#       # self.norm1_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[1], momentum=bn_momentum)
#       # self.block1_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)
#
#       self.final = ME.MinkowskiConvolution(
#           in_channels=TR_CHANNELS[1],
#           out_channels=out_channels,
#           kernel_size=1,
#           stride=1,
#           dilation=1,
#           bias=True,
#           dimension=D)
#
#       self.pruning = ME.MinkowskiPruning()
#
#     def make_layer(self, block_1, block_2, channels, bn_momentum, D):
#       layers = []
#       layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
#       layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
#       layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
#
#       return nn.Sequential(*layers)
#
#     # def get_target_by_sp_tensor(self, out, coords_T):
#     #       with torch.no_grad():
#     #           def ravel_multi_index(coords, step):
#     #               coords = coords.long()
#     #               step = step.long()
#     #               coords_sum = coords[:, 0] \
#     #                         + coords[:, 1]*step \
#     #                         + coords[:, 2]*step*step \
#     #                         + coords[:, 3]*step*step*step
#     #               return coords_sum
#     #
#     #           step = max(out.C.cpu().max(), coords_T.max()) + 1
#     #
#     #           out_sp_tensor_coords_1d = ravel_multi_index(out.C.cpu(), step)
#     #           target_coords_1d = ravel_multi_index(coords_T, step)
#     #
#     #           # test whether each element of a 1-D array is also present in a second array.
#     #           target = np.in1d(out_sp_tensor_coords_1d, target_coords_1d)
#     #
#     #       return torch.Tensor(target).bool()
#     #
#     # def choose_keep(self, out, coords_T, device):
#     # 	with torch.no_grad():
#     # 		feats = torch.from_numpy(np.expand_dims(np.ones(coords_T.shape[0]), 1))
#     # 		x = ME.SparseTensor(features=feats.to(device), coordinates=coords_T.to(device))
#     # 		coords_nums = [len(coords) for coords in x.decomposed_coordinates]
#     #
#     # 		_,row_indices_per_batch = out.coordinate_manager.origin_map(out.coordinate_map_key)
#     # 		keep = torch.zeros(len(out), dtype=torch.bool)
#     # 		for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
#     # 			coords_num = min(len(row_indices), ori_coords_num)# select top k points.
#     # 			values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
#     # 			keep[row_indices[indices]]=True
#     # 	return keep
#
#     def forward(self, x, device, prune=True):
#       print('Mynet forward begin')
#       print_gpu_memory(gpu_index)
#
#       print()
#
#       out_s1 = self.conv1(x)
#       out_s1 = self.norm1(out_s1)
#       out_s1 = self.block1(out_s1)
#       out = MEF.relu(out_s1)
#
#       out_s2 = self.conv2(out)
#       out_s2 = self.norm2(out_s2)
#       out_s2 = self.block2(out_s2)
#       out = MEF.relu(out_s2)
#
#       out_s4 = self.conv3(out)
#       out_s4 = self.norm3(out_s4)
#       out_s4 = self.block3(out_s4)
#       out = MEF.relu(out_s4)
#
#       out_s8 = self.conv4(out)
#       out_s8 = self.norm4(out_s8)
#       out_s8 = self.block4(out_s8)
#       out = MEF.relu(out_s8)
#
#       out_s16 = self.conv5(out)
#       out_s16 = self.norm5(out_s16)
#       out_s16 = self.block5(out_s16)
#       out = MEF.relu(out_s16)
#
#       print('Mynet forward after encoder:')
#       print_gpu_memory(gpu_index)
#
#       out = self.conv5_tr(out)
#       out = self.norm5_tr(out)
#       out = self.block5_tr(out)
#       out_s8_tr = MEF.relu(out)
#
#       out = ME.cat(out_s8_tr, out_s8)
#
#       out = self.conv4_tr(out)
#       out = self.norm4_tr(out)
#       out = self.block4_tr(out)
#       out_s4_tr = MEF.relu(out)
#
#       out = ME.cat(out_s4_tr, out_s4)
#
#       out = self.conv3_tr(out)
#       out = self.norm3_tr(out)
#       out = self.block3_tr(out)
#       out_s2_tr = MEF.relu(out)
#
#       out = ME.cat(out_s2_tr, out_s2)
#
#       out = self.conv2_tr(out)
#       out = self.norm2_tr(out)
#       out = self.block2_tr(out)
#       out_s1_tr = MEF.relu(out)
#
#       out = out_s1_tr + out_s1
#       out = self.conv1_tr(out)
#       out = MEF.relu(out)
#
#       print('Mynet forward after decoder:')
#       print_gpu_memory(gpu_index)
#
#       out_cls = self.final(out)
#       # target = self.get_target_by_sp_tensor(out, pc_data)
#       # keep = self.choose_keep(out_cls, pc_data, device)
#       # if prune:
#       #     out = self.pruning(out_cls, keep.to(device))
#
#       return out_cls #out, out_cls, target, keep

class BasicResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, dimension=3):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class MyNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=3, bn_momentum=0.1, D=3):
        super(MyNet, self).__init__(D)

        # 减少通道数
        CHANNELS = [16, 32, 64, 128]
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=3,
            stride=1,
            bias=False,
            dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(CHANNELS[0], momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)

        # Encoder
        self.encoder1 = self._make_encoder(CHANNELS[0], CHANNELS[1], num_blocks=1, stride=2)
        self.encoder2 = self._make_encoder(CHANNELS[1], CHANNELS[2], num_blocks=1, stride=2)
        self.encoder3 = self._make_encoder(CHANNELS[2], CHANNELS[3], num_blocks=1, stride=2)

        # Bottleneck
        self.bottleneck = BasicResidualBlock(CHANNELS[3], dimension=D)

        # Decoder
        self.decoder3 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[2],
            kernel_size=2,
            stride=2,
            bias=False,
            dimension=D
        )
        self.bn_decoder3 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
        self.decoder_block3 = BasicResidualBlock(CHANNELS[2], dimension=D)

        self.decoder2 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[1],
            kernel_size=2,
            stride=2,
            bias=False,
            dimension=D
        )
        self.bn_decoder2 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)
        self.decoder_block2 = BasicResidualBlock(CHANNELS[1], dimension=D)

        self.decoder1 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[0],
            kernel_size=2,
            stride=2,
            bias=False,
            dimension=D
        )
        self.bn_decoder1 = ME.MinkowskiBatchNorm(CHANNELS[0], momentum=bn_momentum)
        self.decoder_block1 = BasicResidualBlock(CHANNELS[0], dimension=D)

        # Final Convolution
        self.final = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=D
        )

    def _make_encoder(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                bias=False,
                dimension=self.D
            )
        )
        layers.append(ME.MinkowskiBatchNorm(out_channels))
        layers.append(ME.MinkowskiReLU(inplace=True))
        for _ in range(num_blocks):
            layers.append(BasicResidualBlock(out_channels, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder
        dec3 = self.decoder3(bottleneck)
        dec3 = self.bn_decoder3(dec3)
        dec3 = self.relu(dec3)
        dec3 = self.decoder_block3(dec3) + enc2  # 跳跃连接

        dec2 = self.decoder2(dec3)
        dec2 = self.bn_decoder2(dec2)
        dec2 = self.relu(dec2)
        dec2 = self.decoder_block2(dec2) + enc1  # 跳跃连接

        dec1 = self.decoder1(dec2)
        dec1 = self.bn_decoder1(dec1)
        dec1 = self.relu(dec1)
        dec1 = self.decoder_block1(dec1) + x  # 跳跃连接

        # Final Convolution
        out = self.final(dec1)
        return out