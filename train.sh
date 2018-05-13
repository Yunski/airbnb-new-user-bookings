#!/bin/sh

python train.py -m all -s adasyn --device=gpu
# python train.py -m all -s smote --device=gpu
# python train.py -m all -s random --device=gpu
# python train.py -m all -s none --device=gpu

