#!/bin/bash
echo "Making Frame Defintion Embeddings"
python3 ../src/vectorise_defintion.py.py --fn_v $1 --model $2
echo "========================================================"
echo " "
echo "Making AGE resources from the Frame Defintion Embeddings"
python3 ../src/prepareAGEdata.py --fn_v $1 --model $2 
echo "========================================================"
echo "Done"