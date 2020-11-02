import os
import argparse
from tqdm import tqdm
import random
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/bu3dfe_neutral2happiness_n12")
    parser.add_argument("--pairs_list_name", type=str, default="pairs_combine.csv")
    parser.add_argument("--valid_rate", type=float, default=0.10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    domainA_dir = os.path.join( args.dataset_dir, "domainA" )
    domainB_dir = os.path.join( args.dataset_dir, "domainB" )
    domainA_names = sorted( [f for f in os.listdir(domainA_dir) if f.endswith(('.jpg', '.png'))] )
    domainB_names = sorted( [f for f in os.listdir(domainB_dir) if f.endswith(('.jpg', '.png'))] )

    domainAB_names =list( itertools.product(domainA_names, domainB_names) )
    domainAB_names_valid = random.sample(domainAB_names, int(len(domainAB_names) * args.valid_rate))
    if( args.debug ):
        print( "len(domainA_names) : ", len(domainA_names) )
        print( "len(domainB_names) : ", len(domainB_names) )
        print( "len(domainAB_names) : ", len(domainAB_names) )
        print( "len(domainAB_names_valid) : ", len(domainAB_names_valid) )
        print( "domainAB_names[0:5] : ", domainAB_names[0:5] )
        print( "domainAB_names_valid[0:5] : ", domainAB_names_valid[0:5] )

    with open( os.path.join( args.dataset_dir, "train_" + args.pairs_list_name ), "w" ) as f:
        f.write( "domainA_name,domainB_name\n" )
        for domainA_name, domainB_name in tqdm(domainAB_names):
            if( domainA_name != domainB_name ):
                f.write( domainA_name + "," )
                f.write( domainB_name + "\n" )

    with open( os.path.join( args.dataset_dir, "valid_" + args.pairs_list_name ), "w" ) as f:
        f.write( "domainA_name,domainB_name\n" )
        for domainA_name, domainB_name in tqdm(domainAB_names_valid):
            if( domainA_name != domainB_name ):
                f.write( domainA_name + "," )
                f.write( domainB_name + "\n" )