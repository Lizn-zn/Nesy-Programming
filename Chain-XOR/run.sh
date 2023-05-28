# supervised training 

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 20 --k 19 --num_epochs 50000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 20 --exp_name 'parity_20'

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 40 --k 39 --num_epochs 50000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 40 --exp_name 'parity_40'

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 60 --k 59 --num_epochs 50000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 60 --exp_name 'parity_60'

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 80 --k 79 --num_epochs 50000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 80 --exp_name 'parity_80'

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 100 --k 99 --num_epochs 100000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 100 --exp_name 'parity_100'

CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 200 --k 199 --num_epochs 100000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 200 --exp_name 'parity_200'
