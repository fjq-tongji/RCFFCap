

python eval.py --dump_images 0  \
	--num_images 5000  \
        --infos_path log/log_transformer_rl/infos_transformer.pkl \
	--model log/log_transformer_rl/model.pth \
	--language_eval 1  \
	--save_path_seq log/eval_karpathy_test_seq.json \
	--save_path_loss_index log/eval_karpathy_test_loss_index.json

