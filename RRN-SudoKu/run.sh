CUDA_VISIBLE_DEVICES=0 python3 nesy_train_cnn.py --seed 0 --clauses 2000 --exp_name cnn

# CUDA_VISIBLE_DEVICES=1 python3 nesy_train_gpt.py --seed 0 --clauses 2000 --exp_name gpt


#### eval and transfer

# CUDA_VISIBLE_DEVICES=2 python3 nesy_eval_cnn.py --seed 0 --test_root './data/' --file_name 'checkpoint/cnn_model.t7'

# CUDA_VISIBLE_DEVICES=2 python3 nesy_eval_gpt.py --seed 0 --test_root './data/' --file_name 'checkpoint/gpt_model.t7'

# CUDA_VISIBLE_DEVICES=2 python3 nesy_eval_gpt.py --seed 0 --test_root '../SATNet-SudoKu/data' --file_name 'checkpoint/gpt_model.t7'

# CUDA_VISIBLE_DEVICES=1 python3 nesy_eval_cnn.py --seed 0 --test_root '../SATNet-SudoKu/data' --file_name 'checkpoint/cnn_model.t7'
