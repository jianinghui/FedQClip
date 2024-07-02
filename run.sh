#!/bin/bash

# Set default values for parameters
num_clients=4
num_rounds=100
num_epochs_per_round=5
eta_c=0.05
gamma_c=10
eta_s=0.05
gamma_s=1000000
quantize=True
bit=4
flag=True
alpha=1.0
iid=False
batch_size=64
model_name="MobileNetV3"
dataset_name="TinyImagenet"

# Execute the federated learning script with the specified parameters
python federated_learning.py --num_clients $num_clients \
                             --num_rounds $num_rounds \
                             --num_epochs_per_round $num_epochs_per_round \
                             --eta_c $eta_c \
                             --gamma_c $gamma_c \
                             --eta_s $eta_s \
                             --gamma_s $gamma_s \
                             --quantize $quantize \
                             --bit $bit \
                             --flag $flag \
                             --alpha $alpha \
                             --iid $iid \
                             --batch_size $batch_size \
                             --model_name $model_name \
                             --dataset_name $dataset_name
