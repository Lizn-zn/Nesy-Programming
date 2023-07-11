

#### eval and transfer
CUDA_VISIBLE_DEVICES=2 python3 nesy_train.py --game_size 7 --seed 0 --clauses 200 --lamda 1000.0 --num_epochs 500000 
# CUDA_VISIBLE_DEVICES=2 python3 nesy_train.py --game_size 15 --seed 0 --clauses 200 --lamda 1000.0 --num_epochs 500000 

CUDA_VISIBLE_DEVICES=2 python3 nesy_eval.py --game_size 7
# CUDA_VISIBLE_DEVICES=2 python3 nesy_eval.py --game_size 15  



