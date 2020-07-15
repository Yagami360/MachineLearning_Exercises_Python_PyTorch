import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_dataset( dataset_dir, device ):
    #-------------------------------------------------
    # 特徴量（論文中の頻出単語情報と論文カテゴリ）を読み込む
    #-------------------------------------------------
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names =  feature_names + ["subject"]
    df_features = pd.read_csv( os.path.join(dataset_dir, "cora.content"), sep='\t', header=None, names=column_names )
    print( df_features.head() )

    source_ids = df_features.index.values
    features = df_features.iloc[:,0:-1].values
    labels = df_features.iloc[:,-1].values
    #print( "source_ids.shape : ", source_ids.shape )
    #print( "features.shape : ", features.shape )
    #print( "labels.shape : ", labels.shape )
    #print( "source_ids.dtype : ", source_ids.dtype )
    #print( "features.dtype : ", features.dtype )
    #print( "labels.dtype : ", labels.dtype )

    # label 値を数値化
    """
    encoder = OneHotEncoder()
    labels = encoder.fit_transform( labels.reshape(1,-1) )
    """
    labels = encode_onehot(labels)
    #print( "labels.shape : ", labels.shape )
    #print( "labels.dtype : ", labels.dtype )

    # 特徴量を疎行列化
    features = sp.csr_matrix( features, dtype=np.float32)

    # 特徴量を正規化
    features = normalize(features)

    #-------------------------------------------------
    # 隣接行列情報を読み込む
    #-------------------------------------------------
    df_edges = pd.read_csv( os.path.join(dataset_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"] )
    print( df_edges.head() )

    source_ids_map = {j: i for i, j in enumerate(source_ids)}
    #print( "source_ids_map :", source_ids_map )

    edges = df_edges.values
    #print( "edges.flatten() : ", edges.flatten() )
    edges = np.array( list(map(source_ids_map.get, edges.flatten())), dtype=np.int32 ).reshape(edges.shape)
    #print( "edges.shape : ", edges.shape )

    # 辺情報から隣接行列の疎行列を作成
    adj_matrix = sp.coo_matrix( (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32 )

    # build symmetric adjacency matrix
    adj_matrix = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)

    # 隣接行列を正規化
    adj_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

    #-------------------------------------------------
    # PyTorch 型に変換
    #-------------------------------------------------
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix).to(device)
    return features, labels, adj_matrix