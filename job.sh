#!/bin/bash
module load anaconda3
source activate jacobenv
declare -a tasks=("Breakout-v0" "BeamRider-v0" "Enduro-v0" "Qbert-v0" "Seaquest-v0" "SpaceInvaders-v0")
declare optim="SGD"
declare batch=32

for task in ${tasks[@]}
do
	python main.py -task=$task -batch=32 -optim=$optim -lr=0.01
done

for task in ${tasks[@]}
do
	python main.py -task=$task -batch=32 -optim=$optim -lr=0.00001
done