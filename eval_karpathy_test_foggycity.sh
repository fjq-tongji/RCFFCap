python eval.py --dump_images 0  \
	--num_images 1525  \
        --infos_path log/log_foggycity_1016_nic/infos_show_tell.pkl \
	--model log/log_foggycity_1016_nic/model.pth \
	--language_eval 1  \
	--beam_size  2     \
	--batch_size 100   \
	--split test       \
	--save_path_seq log/log_foggycity_1016_nic/eval_karpathy_test_seq_coco.json \
	--save_path_loss_index log/log_foggycity_1016_nic/eval_karpathy_test_loss_index_coco.json  \

