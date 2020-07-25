# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image

#=======================================
# Geometric transform 関連
#=======================================
class ImagePadSymmetric(nn.Module):
    """
    左右上下対称画像（シンメトリック）でパディングした画像を生成 / 画像サイズ２倍
    generate symmetrically padded image for bigger sampling region
    """
    def __init__( self, device = torch.device("cuda"), padding_factor = 0.5 ):
        super(ImagePadSymmetric, self).__init__()
        self.device = device
        self.padding_factor = padding_factor
        return

    def forward( self, image ):
        b, c, h, w = image.size()
        pad_h, pad_w = int(h * self.padding_factor), int(w * self.padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1)).to(self.device)
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1)).to(self.device)
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1)).to(self.device)
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1)).to(self.device)
        #print( "pad_h={}, pad_w={}".format(pad_h,pad_w) )
        #print( "idx_pad_left={}, idx_pad_right={}, idx_pad_top={}, idx_pad_bottom={}".format(idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom ) )

        # torch.index_select() : 
        image = torch.cat( [
            image.index_select(3,idx_pad_left),
            image,
            image.index_select(3,idx_pad_right)
        ], 3 )

        image = torch.cat( [
            image.index_select(2,idx_pad_top),
            image,
            image.index_select(2,idx_pad_bottom)
        ], 2 )

        #print( "image.shape : ", image.shape )
        return image


class AffineTransform( nn.Module ):
    """
    affine 変換
    """
    def __init__( 
        self, 
        image_height = 240, image_width = 240, n_out_channels = 3, 
        padding_mode = "border",
        offset_factor = 1.0, padding_factor = 1.0, crop_factor = 1.0,
    ):
        super(AffineTransform, self).__init__()        
        self.image_height = image_height
        self.image_width = image_width
        self.n_out_channels = n_out_channels
        self.padding_mode = padding_mode
        self.offset_factor = offset_factor
        self.padding_factor = padding_factor
        self.crop_factor = crop_factor

        self.theta_identity = torch.Tensor( np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32) )
        return
        
    def forward(self, image, theta = None ):
        """
        [Args]
            image : <Tensor> 変換対象画像
            theta : <Tensor> affine 変換の変換パラメータ / shape = [B, 6]
        """
        if theta is None:
            theta = self.theta_identity
            theta = theta.expand(image.shape[0],2,3).contiguous()
            theta = Variable(theta, requires_grad=False)
        if not theta.shape == (theta.shape[0],2,3):
            theta = theta.view(-1,2,3).contiguous()

        #----------------------
        # Affine 変換
        #----------------------
        # torch.Size() :  torch.Size テンソルを取得 / torch.Size([B, n_out_channels, H, W])
        out_size = torch.Size( (theta.shape[0], self.n_out_channels, self.image_height, self.image_width) )
        grid = F.affine_grid( theta, out_size )

        #-------------------------------
        # sampling_grid から変換画像を生成
        #-------------------------------
        # rescale grid according to crop_factor and padding_factor
        if( self.padding_factor != 1 or self.crop_factor != 1 ):
            grid = grid * (self.padding_factor * self.crop_factor)
        # rescale grid according to offset_factor
        if( self.offset_factor != 1 ):
            grid = grid * self.offset_factor

        warp_image = F.grid_sample(image, grid, padding_mode = self.padding_mode )
        return warp_image, grid


class TpsTransform(nn.Module):
    """
    TPS 変換
    """
    def __init__(
        self, device = torch.device("cuda"), 
        image_height = 240, image_width = 240,
        use_regular_grid = True, grid_size = 3, reg_factor = 0, padding_mode = "border",
        offset_factor = 1.0, padding_factor = 1.0, crop_factor = 1.0,
    ):
        super(TpsTransform, self).__init__()
        self.device = device
        self.image_height, self.image_width = image_height, image_width
        self.reg_factor = reg_factor
        self.padding_mode = padding_mode
        self.offset_factor = offset_factor
        self.padding_factor = padding_factor
        self.crop_factor = crop_factor

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

        self.theta_identity = torch.Tensor( np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32) )
        return

    def forward(self, image, theta = None ):
        if theta is None:
            theta = self.theta_identity
            theta = theta.expand(image.shape[0],2,3).contiguous()
            theta = Variable(theta, requires_grad=False)

        #----------------------
        # TPS 変換
        #----------------------
        grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))        

        #-------------------------------
        # sampling_grid から変換画像を生成
        #-------------------------------
        # rescale grid according to crop_factor and padding_factor
        if( self.padding_factor != 1 or self.crop_factor != 1 ):
            grid = grid * (self.padding_factor * self.crop_factor)
        # rescale grid according to offset_factor
        if( self.offset_factor != 1 ):
            grid = grid * self.offset_factor

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


class PointTranform(nn.Module):
    """
    座標点 (x,y) を幾何学的変換モデルで変換する
    Class with functions for transforming a set of points with affine/tps transformations    
    """
    def __init__(
        self, device = torch.device("cuda"),
        image_height = 240, image_width = 240,
        tps_grid_size = 3, tps_reg_factor = 0
    ):
        super(PointTranform, self).__init__()
        self.device=device
        self.tps_transform = TpsTransform( device = device, image_height = image_height, image_width = image_width, grid_size = tps_grid_size, reg_factor = tps_reg_factor )   
        return    

    def homography_mat_from_4_pts( self, theta ):
        b=theta.size(0)
        if not theta.size()==(b,8):
            theta = theta.view(b,8)
            theta = theta.contiguous()
        
        xp=theta[:,:4].unsqueeze(2) ;yp=theta[:,4:].unsqueeze(2) 
        x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
        y = Variable(torch.FloatTensor([-1,  1,-1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
        z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
        o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
        single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b,1,1)
        
        if theta.is_cuda:
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            o = o.cuda()
            single_o = single_o.cuda()

        A=torch.cat([torch.cat([-x,-y,-o,z,z,z,x*xp,y*xp,xp],2),torch.cat([z,z,z,-x,-y,-o,x*yp,y*yp,yp],2)],1)
        # find homography by assuming h33 = 1 and inverting the linear system
        h=torch.bmm(torch.inverse(A[:,:,:8]),-A[:,:,8].unsqueeze(2))
        # add h33
        h=torch.cat([h,single_o],1)
        H = h.squeeze(2)
        return H

    def transform_affine( self, theta, points ):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2],points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        return warped_points

    def transform_tps( self, theta, points ):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points=points.unsqueeze(3).transpose(1,3)
        # apply transformation
        warped_points = self.tps_transform.apply_transformation(theta,points)
        # undo reshaping
        warped_points=warped_points.transpose(3,1).squeeze(3)      
        return warped_points

    def transform_hom( self, theta, points, eps = 1e-5 ):
        b=theta.size(0)
        if theta.size(1)==9:
            H = theta            
        else:
            H = self.homography_mat_from_4_pts(theta)            

        h0=H[:,0].unsqueeze(1).unsqueeze(2)
        h1=H[:,1].unsqueeze(1).unsqueeze(2)
        h2=H[:,2].unsqueeze(1).unsqueeze(2)
        h3=H[:,3].unsqueeze(1).unsqueeze(2)
        h4=H[:,4].unsqueeze(1).unsqueeze(2)
        h5=H[:,5].unsqueeze(1).unsqueeze(2)
        h6=H[:,6].unsqueeze(1).unsqueeze(2)
        h7=H[:,7].unsqueeze(1).unsqueeze(2)
        h8=H[:,8].unsqueeze(1).unsqueeze(2)

        X=points[:,0,:].unsqueeze(1)
        Y=points[:,1,:].unsqueeze(1)
        Xp = X*h0+Y*h1+h2
        Yp = X*h3+Y*h4+h5
        k = X*h6+Y*h7+h8
        # prevent division by 0
        k = k+torch.sign(k)*eps
        Xp /= k; Yp /= k
        return torch.cat((Xp,Yp),1)
    

class GeoPadTransform(nn.Module):
    """
    幾何変換モデルを用いて、theta_gt から目標画像（＝変形画像）を生成。
    幾何変換モデルによって生じる border effect を防ぐため、参照画像画像周辺に padding 処理も行う
    """
    def __init__(
        self, device = torch.device("cuda"),
        image_height = 240, image_width = 240,
        geometric_model = "affine",
        padding_factor = 0.5, crop_factor = 9/16, occlusion_factor = 0.0,
    ):
        super(GeoPadTransform, self).__init__()
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.geometric_model = geometric_model
        self.padding_factor = padding_factor
        self.crop_factor = crop_factor
        self.occlusion_factor = occlusion_factor

        # 
        self.image_pad_sym = ImagePadSymmetric(self.device, self.padding_factor)

        # 参照画像の crop 処理用アフィン変換
        self.affine_transform = AffineTransform( image_height = self.image_height, image_width = self.image_width, n_out_channels = 3, padding_mode = "border", padding_factor = self.padding_factor, crop_factor = self.crop_factor )

        # 参照画像を変形する幾何学的変換モデル
        if( geometric_model == "affine" ):
            self.geo_transform = AffineTransform( image_height = self.image_height, image_width = self.image_width, n_out_channels = 3, padding_mode = "border", padding_factor = self.padding_factor, crop_factor = self.crop_factor )
        elif( geometric_model == "tps" ):
            self.geo_transform = TpsTransform(device = self.device, image_height = self.image_height, image_width = self.image_width, use_regular_grid = True, grid_size = 3, reg_factor = 0, padding_mode = "border", padding_factor = self.padding_factor, crop_factor = self.crop_factor )
        else:
            NotImplementedError()

        # [ToDO] crop_factor != 0 時の処理 / AffineGridGenV2
        pass

        return

    def forward( self, image_s, theta_gt ):
        #---------------------------------------------------------------------------
        # 幾何変換によって生じる border effect を防ぐため、参照画像画像周辺に padding 処理
        #---------------------------------------------------------------------------
        # symmetrically image padding で大きいサイズの参照画像を取得
        image_s = self.image_pad_sym(image_s)

        # convert to variables
        image_s = Variable(image_s, requires_grad = False )
        theta_gt =  Variable(theta_gt, requires_grad = False )        

        # affine 変換で参照画像をクロップ
        image_s_crop, _ = self.affine_transform(image_s, None)

        #---------------------------------------------------------------------------
        # 幾何学的変換モデルで、変換画像を生成
        #---------------------------------------------------------------------------
        # 目標画像を取得 / 参照画像を theta_gt で変形
        image_t, grid_t = self.geo_transform( image_s, theta_gt )

        # [ToDO] occlusion_factor !=0 時の処理
        pass

        return image_s_crop, image_t, grid_t
