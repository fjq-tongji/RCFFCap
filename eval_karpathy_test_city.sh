python eval.py --dump_images 0  \
	--num_images 1525  \
        --infos_path log/log_transformer_city_0920_LW/infos_transformer_lw.pkl \
	--model log/log_transformer_city_0920_LW/model.pth \
	--language_eval 1  \
	--beam_size  2     \
	--batch_size 100   \
	--split test       \
	--save_path_seq log/log_transformer_city_0920_LW/eval_karpathy_test_seq_coco.json \
	--save_path_loss_index log/log_transformer_city_0920_LW/eval_karpathy_test_loss_index_coco.json  \

