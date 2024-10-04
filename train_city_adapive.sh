id="transformer_city_adaptive"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train_2.py --id adaatt  \
    --caption_model adaatt  \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type LSTM \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json data/cityscapes/cityscapes_paragraph_trainval_talk.json \
    --input_label_h5 data/cityscapes/cityscapes_paragraph_trainval_talk.h5 \
    --input_fc_dir  data/cityscapes/cityscapes_fc \
    --input_att_dir  data/cityscapes/cityscapes_att  \
    --input_word_dir /data2/fjq/1.next_paper/Cityscapes-total/5.city_label_filter_1.0/num  \
    --input_attr_dir data/cityscapes/cityscapes_attr_name/num    \
    --input_seg_dir  /data2/fjq/1.next_paper/Cityscapes-total/7.city_seg_filter_0.8/num  \
    --input_box_dir  data/cityscapes/cityscapes_box      \
    --seq_per_img 1 \
    --batch_size 32 \
    --beam_size 2 \
    --learning_rate 5e-4 \
    --num_layers 4 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --att_hid_size  1024  \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from \
    --save_checkpoint_every 200 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 20 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3
