# supervised training 

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 20 --k 19 --b 20 --clauses 1000 --num_epochs 200000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 20 --k 19 --b 2 --clauses 1000 --num_epochs 500000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 20 --exp_name 'conj_20'

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 40 --k 39 --b 40 --clauses 1000 --num_epochs 200000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 40 --k 39 --b 2 --clauses 1000 --num_epochs 500000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 40 --exp_name 'conj_40'

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 60 --k 59 --b 60 --clauses 1000 --num_epochs 200000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 60 --k 59 --b 2 --clauses 1000 --num_epochs 500000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 60 --exp_name 'conj_60'

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 80 --k 79 --b 80 --clauses 1000 --num_epochs 500000 
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 80 --k 79 --b 2 --clauses 1500 --num_epochs 1000000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 80 --exp_name 'conj_80'

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 100 --k 99 --b 100 --clauses 1000 --num_epochs 500000 
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 100 --k 99 --b 2 --clauses 2000 --num_epochs 1000000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 100 --exp_name 'conj_100'

# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 0.0 --len 200 --k 199 --b 200 --clauses 1000 --num_epochs 500000
# CUDA_VISIBLE_DEVICES=4,5 python3 nesy_train.py --lamda 1000.0 --len 200 --k 199 --b 2 --clauses 3000 --num_epochs 1000000
CUDA_VISIBLE_DEVICES=4,5 python3 nesy_eval.py --len 200 --exp_name 'conj_200'
