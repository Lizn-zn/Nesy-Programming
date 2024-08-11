# supervised training
# CUDA_VISIBLE_DEVICES=4,5 python3 sup_train.py --seed 0 --alpha 0.75 --batch_size 32

# nesy learning

# CUDA_VISIBLE_DEVICES=0 python3 nesy_train_cnn.py --seed 0 --clauses 3000 --exp_name 'logic_clauses_3000' --iter 30  --num_epochs 300

CUDA_VISIBLE_DEVICES=0 python3 nesy_train_gpt.py --seed 0 --clauses 3000 --exp_name 'gpt_logic_clauses_3000' --iter 30  --num_epochs 300


# CUDA_VISIBLE_DEVICES=0 python3 nesy_eval.py --seed 0 --conf_threshold 0.9995 --test_root './data' --file_name 'checkpoint/lenet_0_nesy_programming_0.t7'

# CUDA_VISIBLE_DEVICES=0 python3 nesy_eval.py --seed 0 --conf_threshold 0.0 --test_root '../RRN-SudoKu/data' --file_name 'checkpoint/lenet_0_nesy_programming_0.t7'

# CUDA_VISIBLE_DEVICES=0 python3 nesy_eval.py --seed 0 --conf_threshold 0.9995 --test_root './data' --file_name 'checkpoint/lenet_0_nesy_sgd_0.t7'