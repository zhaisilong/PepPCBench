#!/bin/bash
set -e

mmseqs easy-cluster chains.fasta clusterRes tmp_mmseqs --min-seq-id 0.4 -c 0.8
