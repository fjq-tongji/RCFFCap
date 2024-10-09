id="transformer_bushiyan_SRFNet_wanzhengban_Small"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train_2.py --id transformer_3  \
    --caption_model transformer_3 \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 6 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json data/coco/cocotalk.json \
    --input_label_h5 data/coco/cocotalk_label.h5 \
    --input_fc_dir  /data/fjq/1.next_paper/cocobu_fc \
    --input_att_dir  /data/fjq/1.next_paper/cocobu_att  \
    --input_word_dir /data/fjq/1.next_paper/clip_retrieval_filter_new/coco_label_filter_results/cocobu_label_filter_1.0_res101/num  \
    --input_attr_dir /data/fjq/1.next_paper/clip_retrieval_filter_new/coco_attr_filter_results/cocobu_attr_filter_1.0_res101/num   \
    --input_seg_dir /data/fjq/1.next_paper/clip_retrieval_filter_new/coco_seg_filter_results/cocobu_seg_filter_0.8_res101/num  \
    --input_box_dir  /data/fjq/1.next_paper/cocobu_box      \
    --seq_per_img 5 \
    --batch_size 48  \
    --beam_size 2 \
    --use_box   1  \
    --learning_rate 5e-4 \
    --num_layers 4 \
    --input_encoding_size 96 \
    --rnn_size 384 \
    --learning_rate_decay_start 0 \
    --learning_rate_decay_rate 0.8 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from    \
    --save_checkpoint_every 2000 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 15 \
    --save_history_ckpt 0   \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3
