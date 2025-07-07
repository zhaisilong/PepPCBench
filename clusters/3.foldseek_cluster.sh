#!/bin/bash

set -e

foldseek easy-multimercluster pdb_train_test_data clu tmp --min-seq-id 0.4 -c 0.8 --gpu 1 --cluster-mode 0