import os
import sys
import argparse
from tqdm import tqdm
import random
import itertools

sys.path.append(os.path.join(os.getcwd(), '../'))
from utils.utils import numerical_sort

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/zalando_dataset_n20")
    parser.add_argument("--pairs_list_name", type=str, default="pairs.csv")
    parser.add_argument("--valid_rate", type=float, default=0.50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    pose_dir = os.path.join( args.dataset_dir, "pose" )
    pose_parsing_dir = os.path.join( args.dataset_dir, "pose_parsing" )
    pose_names = sorted( [f for f in os.listdir(pose_dir) if f.endswith(('.jpg', '.png'))], key = numerical_sort )
    pose_parsing_names = sorted( [f for f in os.listdir(pose_parsing_dir) if f.endswith(('.jpg', '.png'))], key = numerical_sort )

    random.seed(args.seed)
    pose_names_valid = sorted( random.sample(pose_names, int(len(pose_names) * args.valid_rate)), key = numerical_sort )
    random.seed(args.seed)
    pose_parsing_names_valid = sorted( random.sample(pose_parsing_names, int(len(pose_parsing_names) * args.valid_rate)), key = numerical_sort )
    pose_names = list( set(pose_names) - set(pose_names_valid) )
    pose_parsing_names = list( set(pose_parsing_names) - set(pose_parsing_names_valid) )
    if( args.debug ):
        print( "len(pose_names) : ", len(pose_names) )
        print( "len(pose_parsing_names) : ", len(pose_parsing_names) )
        print( "len(pose_parsing_names) : ", len(pose_parsing_names) )
        print( "len(pose_names_valid) : ", len(pose_names_valid) )
        print( "pose_parsing_names[0:5] : ", pose_parsing_names[0:5] )

    with open( os.path.join( args.dataset_dir, "train_" + args.pairs_list_name ), "w" ) as f:
        f.write( "pose_parsing_name,pose_name\n" )
        for pose_name, pose_parsing_names in zip(pose_names, pose_parsing_names):
            f.write( pose_parsing_names + "," )
            f.write( pose_name + "\n" )

    with open( os.path.join( args.dataset_dir, "valid_" + args.pairs_list_name ), "w" ) as f:
        f.write( "pose_parsing_name,pose_name\n" )
        for pose_name, pose_parsing_names in zip(pose_names_valid, pose_parsing_names_valid):
            f.write( pose_parsing_names + "," )
            f.write( pose_name + "\n" )