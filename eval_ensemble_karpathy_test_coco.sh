

python eval_ensemble.py 
	--ids transformer1,transformer2,transformer3,transformer4      \
	--weights 0.25,0.25,0.25,0.25     \
	--batch_size 100   \
	--dump_images 0  \
	--num_images 5000  \
	--split test       \
	--language_eval 1  \
	--beam_size  2     \
	--temperature 1.0   \
	--sample_method greedy   \
	--max_length  30  \
	--save_path_seq log/log_transformer/eval_karpathy_test_seq_coco.json \
	--save_path_loss_index log/log_transformer/eval_karpathy_test_loss_index_coco.json

