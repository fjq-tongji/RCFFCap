python eval.py --dump_images 0  \
	--num_images 5000  \
        --infos_path log/log_transformer_new_att_keshihua_0913/infos_transformer_1.pkl \
	--model log/log_transformer_new_att_keshihua_0913/model.pth \
	--language_eval 1  \
	--beam_size  2     \
	--batch_size 2   \
	--split test       \
	--save_path_seq log/log_transformer_new_att_keshihua_0913/eval_karpathy_test_seq_coco.json \
	--save_path_loss_index log/log_transformer_new_att_keshihua_0913/eval_karpathy_test_loss_index_coco.json  \

