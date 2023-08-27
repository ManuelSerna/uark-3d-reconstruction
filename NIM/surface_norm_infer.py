import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from utils.base_parser import get_base_parser


class NIM(nn.Module):
    """Normal Inference Module"""

    def __init__(self):
        super(NIM, self).__init__()

    def forward(self, depth:torch.tensor=None, calib=None, sign_filter=None):
        """Generate surface normal estimation from depth images

        Args:
            depth (torch.Tensor): depth image
            calib (CalibInfo): calibration parameters
            sign_filter (bool): if True, our NIM will additionally utilize a sign filter

        Returns:
            torch.Tensor: surface normal estimation
        """
        camParam = torch.tensor([
            [calib['focal'], 0.0,            calib['u']], 
            [0.0,            calib['focal'], calib['v']], 
            [0.0,            0.0,            1.0]
        ])

        h, w = depth.size()
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        v_map = v_map.type(torch.float32)
        u_map = u_map.type(torch.float32)

        Z = depth   # h, w
        Y = Z * (v_map - camParam[1, 2]) / camParam[0, 0]  # h, w
        X = Z * (u_map - camParam[0, 2]) / camParam[0, 0]  # h, w
        Z[Y <= 0] = 0
        Y[Y <= 0] = 0
        Z[torch.isnan(Z)] = 0
        D = torch.ones(h, w) / Z  # h, w

        Gx = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                          dtype=torch.float32)
        Gy = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]],
                          dtype=torch.float32)

        Gu = F.conv2d(D.view(1, 1, h, w), Gx.view(1, 1, 3, 3), padding=1)
        Gv = F.conv2d(D.view(1, 1, h, w), Gy.view(1, 1, 3, 3), padding=1)

        nx_t = Gu * camParam[0, 0]   # 1, 1, h, w
        ny_t = Gv * camParam[1, 1]   # 1, 1, h, w

        phi = torch.atan(ny_t / nx_t) + torch.ones([1, 1, h, w]) * 3.141592657
        a = torch.cos(phi)
        b = torch.sin(phi)

        # was this covered in the paper?
        diffKernelArray = torch.tensor([[0, -1, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, -1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, -1, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, -1, 0]], dtype=torch.float32)

        nx_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)
        ny_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)
        nz_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)

        for i in range(4):
            diffKernel = diffKernelArray[i].view(1, 1, 3, 3)
            X_d = F.conv2d(X.view(1, 1, h, w), diffKernel, padding=1)
            Y_d = F.conv2d(Y.view(1, 1, h, w), diffKernel, padding=1)
            Z_d = F.conv2d(Z.view(1, 1, h, w), diffKernel, padding=1)

            nz_i = -(nx_t * X_d + ny_t * Y_d) / Z_d
            norm = torch.sqrt(nx_t * nx_t + ny_t * ny_t + nz_i * nz_i)
            nx_t_i = nx_t / norm
            ny_t_i = ny_t / norm
            nz_t_i = nz_i / norm

            nx_t_i[torch.isnan(nx_t_i)] = 0
            ny_t_i[torch.isnan(ny_t_i)] = 0
            nz_t_i[torch.isnan(nz_t_i)] = 0

            nx_volume[0, i, :, :] = nx_t_i
            ny_volume[0, i, :, :] = ny_t_i
            nz_volume[0, i, :, :] = nz_t_i

        if sign_filter:
            nz_volume_pos = torch.sum(nz_volume > 0, dim=1, keepdim=True)
            nz_volume_neg = torch.sum(nz_volume < 0, dim=1, keepdim=True)
            pos_mask = (nz_volume_pos >= nz_volume_neg) * (nz_volume > 0)
            neg_mask = (nz_volume_pos < nz_volume_neg) * (nz_volume < 0)
            final_mask = pos_mask | neg_mask
            nx_volume *= final_mask
            ny_volume *= final_mask
            nz_volume *= final_mask

        theta = torch.atan((torch.sum(nx_volume, 1) * a +
                            torch.sum(ny_volume, 1) * b) / torch.sum(nz_volume, 1))
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)

        nx[torch.isnan(nz)] = 0
        ny[torch.isnan(nz)] = 0
        nz[torch.isnan(nz)] = -1

        sign_map = torch.ones((1, 1, h, w), dtype=torch.float32)
        sign_map[ny > 0] = -1

        nx = (nx * sign_map).squeeze(dim=0)
        ny = (ny * sign_map).squeeze(dim=0)
        nz = (nz * sign_map).squeeze(dim=0)

        return torch.cat([nx, ny, nz], dim=0)


def normal_visualization(normal:np.array=None):
    """ Normalize surface normals Numpy array to visualize."""
    normal_vis = (1 + normal) / 2
    return normal_vis


def read_depth_img(filepath:str = None):
    """ Read depth image file.
    
    Return:
        Torch.tensor
    """
    depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
    depth = torch.tensor(depth)
    return depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser("NIM Demo", parents=[get_base_parser()])
    args = parser.parse_args()
    
    depth_filepath = os.path.join(args.depth)
    depth = read_depth_img(depth_filepath)

    model = NIM()
    
    calib = {
        'focal': args.focal, # focal for x and y
        'u': depth.shape[1] / 2, # principal pt x component (width/2)
        'v': depth.shape[0] * 0.0, # principal pt y component (height/2)
    }

    normal = model(depth, calib, sign_filter=True)
    normal = normal.cpu().numpy()

    normal_vis = normal_visualization(normal)
    
    '''if not os.path.exists(os.path.join('examples', 'normal')):
        os.makedirs(os.path.join('examples', 'normal'))
    cv2.imwrite(os.path.join('examples', 'normal', example_name + '.png'), cv2.cvtColor(
        normal_vis.transpose([1, 2, 0])*255, cv2.COLOR_RGB2BGR))
    '''
    cv2.imwrite(
        "out.png",
        cv2.cvtColor(normal_vis.transpose([1, 2, 0])*255, cv2.COLOR_RGB2BGR)
    )    

