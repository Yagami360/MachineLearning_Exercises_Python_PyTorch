#!/bin/sh
set -eu

cd ../dataset/
wget -c https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
tar -xvf maps.tar.gz
rm -r maps.tar.gz