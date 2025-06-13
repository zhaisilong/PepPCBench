#!/bin/bash

NATIVE_PDB=$1
PREDICTED_PDB=$2
WORK_DIR=$(dirname $1)
JSON_FILE_NAME=$3  # dockq_relaxed.json or dockq.json

pushd $WORK_DIR > /dev/null
# echo "Running DockQ: DockQ $NATIVE_PDB $PREDICTED_PDB --capri_peptide --json dockq.json --short"
DockQ $NATIVE_PDB $PREDICTED_PDB --capri_peptide --json $JSON_FILE_NAME --short --mapping AB:AB 2>&1 > /dev/null
# echo "json file: $WORK_DIR/dockq.json"
popd > /dev/null
