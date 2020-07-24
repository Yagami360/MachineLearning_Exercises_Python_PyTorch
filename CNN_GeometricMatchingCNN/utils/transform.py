# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

class ImagePadSymmetric(object):
    """
    左右上下対称画像（シンメトリック）でパディングした画像を生成 / 画像サイズ２倍
    generate symmetrically padded image for bigger sampling region
    """
    def __init__( self, device = torch.device("cuda"), padding_factor = 0.5 ):
        self.device = device
        self.padding_factor = padding_factor
        return

    def __call__( self, image_batch ):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * self.padding_factor), int(w * self.padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1)).to(self.device)
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1)).to(self.device)
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1)).to(self.device)
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1)).to(self.device)
        #print( "pad_h={}, pad_w={}".format(pad_h,pad_w) )
        #print( "idx_pad_left={}, idx_pad_right={}, idx_pad_top={}, idx_pad_bottom={}".format(idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom ) )

        # torch.index_select() : 
        image_batch = torch.cat( [
            image_batch.index_select(3,idx_pad_left),
            image_batch,
            image_batch.index_select(3,idx_pad_right)
        ], 3 )

        image_batch = torch.cat( [
            image_batch.index_select(2,idx_pad_top),
            image_batch,
            image_batch.index_select(2,idx_pad_bottom)
        ], 2 )

        #print( "image_batch.shape : ", image_batch.shape )
        return image_batch


class AffineTransform( nn.Module ):
    """
    affine 変換
    """
    def __init__( self, image_height = 240, image_width = 240, n_out_channels = 3, padding_mode = "border" ):
        super(AffineTransform, self).__init__()        
        self.image_height = image_height
        self.image_width = image_width
        self.n_out_channels = n_out_channels
        self.padding_mode = padding_mode
        return
        
    def forward(self, image, theta):
        """
        [Args]
            image : <Tensor> 変換対象画像
            theta : <Tensor> affine 変換の変換パラメータ / shape = [B, 6]
        """
        if not theta.shape == (theta.shape[0],2,3):
            theta = theta.view(-1,2,3).contiguous()

        #  torch.Size() :  torch.Size テンソルを取得 / torch.Size([B, n_out_channels, H, W])
        out_size = torch.Size( (theta.shape[0], self.n_out_channels, self.image_height, self.image_width) )
        grid = F.affine_grid( theta, out_size )
        warp_image = F.grid_sample(image, grid, padding_mode = self.padding_mode )
        return warp_image, grid


from torch.autograd import Variable

class TpsTransform(nn.Module):
    """
    TPS 変換
    """
    def __init__(self, device = torch.device("cuda"), image_height = 240, image_width = 240, use_regular_grid = True, grid_size = 3, reg_factor = 0, padding_mode = "border" ):
        super(TpsTransform, self).__init__()
        self.device = device
        self.image_height, self.image_width = image_height, image_width
        self.reg_factor = reg_factor
        self.padding_mode = padding_mode

        # create grid in numpy
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,image_width),np.linspace(-1,1,image_height))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X,requires_grad=False).to(self.device)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False).to(self.device)

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = Variable(self.compute_L_inverse(P_X,P_Y).unsqueeze(0),requires_grad=False)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_X = Variable(self.P_X,requires_grad=False).to(self.device)
            self.P_Y = Variable(self.P_Y,requires_grad=False).to(self.device)
            
    def forward(self, image, theta):
        grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))        
        warp_image = F.grid_sample(image, grid, padding_mode = self.padding_mode )
        return warp_image, grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K+=torch.eye(K.size(0),K.size(1))*self.reg_factor
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L).to(self.device)
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)