#!/bin/bash
bash ./prepare.sh xp_$1
python3 ../src/vectorise.py --fn_v $1 --model_name $2 --model_path $3
