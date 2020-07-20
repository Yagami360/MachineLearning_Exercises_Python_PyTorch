import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch

# ラベル変換マップ
map_graphonomy_idx_to_rgb = {
    0 : (0,0,0), 1 : (128,0,0), 2 : (255,0,0), 3 : (0,85,0), 4 : (170,0,51),
    5 : (255,85,0), 6 : (0,0,85), 7 : (0,119,221), 8 : (85,85,0), 9 : (0,85,85),
    10 : (85,51,0), 11 : (52,86,128), 12 : (0,128,0), 13 : (0,0,255), 14 : (51,170,221),
    15 : (0,255,255), 16 : (85,255,170), 17 : (170,255,85), 18 : (255,255,0), 19 : (255,170,0),
    20 : (0,255,65), 21 : (83,53,99), 22 : (255,228,225), 23 : (139,125,123), 24 : (188,143,143),
}

map_pascal_idx_to_rgb = {
    0 : [0, 0, 0], 1 : [128, 0, 0], 2 : [0, 128, 0], 3 : [128, 128, 0], 4 : [0, 0, 128],
    5 : [128, 0, 128], 6 : [0, 128, 128], 7 : [128, 128, 128], 8 : [64, 0, 0], 9 : [192, 0, 0],
    10 : [64, 128, 0], 11 : [192, 128, 0], 12 : [64, 0, 128], 13 : [192, 0, 128], 14 : [64, 128, 128],
    15 : [192, 128, 128], 16 : [0, 64, 0], 17 : [128, 64, 0], 18 : [0, 192, 0], 19 : [128, 192, 0],
    20 : [0, 64, 128],
}


def decode_labels_tsr( semantics, dataset_type = "pascal" ):
    """
    [Args]
        semantics : <Tensor> shape = [B,H,W]
    """
    if( dataset_type == "pascal" ):
        n_classes = 21
        map_idx_to_rgb = map_pascal_idx_to_rgb
    else:
        raise NotImplementedError

    semantics_np = semantics.detach().cpu().numpy()

    # batch でループ
    semantic_np_rgbs = []
    for semantic_np in semantics_np:        
        r = semantic_np.copy()
        g = semantic_np.copy()
        b = semantic_np.copy()
        for i in range(0, n_classes):
            r[semantic_np == i] = map_idx_to_rgb[i][0]
            g[semantic_np == i] = map_idx_to_rgb[i][1]
            b[semantic_np == i] = map_idx_to_rgb[i][2]

        semantic_rgb_np = np.zeros((semantic_np.shape[0], semantic_np.shape[1], 3))     # (H,W,3)
        semantic_rgb_np[:, :, 0] = r / 255.0
        semantic_rgb_np[:, :, 1] = g / 255.0
        semantic_rgb_np[:, :, 2] = b / 255.0

        semantic_np_rgbs.append(semantic_rgb_np)

    semantic_rgbs = torch.from_numpy(np.array(semantic_np_rgbs).transpose([0, 3, 1, 2]))
    return semantic_rgbs
