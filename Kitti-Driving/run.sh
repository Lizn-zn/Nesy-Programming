# supervised training
# CUDA_VISIBLE_DEVICES=4,5 python3 sup_train.py --seed 0 --alpha 0.75 --batch_size 32

# nesy learning

# CUDA_VISIBLE_DEVICES=0 python3 nesy_train_cnn.py --seed 0 --clauses 3000 --exp_name 'logic_clauses_3000' --iter 30  --num_epochs 300

# CUDA_VISIBLE_DEVICES=0 python3 nesy_train_gpt.py --seed 0 --clauses 3000 --exp_name 'gpt_logic_clauses_3000' --iter 30  --num_epochs 300


