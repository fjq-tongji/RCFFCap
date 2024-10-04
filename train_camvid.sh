id="transformer_camvid_0917_att"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train_2.py --id transformer_1  \
    --caption_model transformer_1 \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json data/camvid/CamVid_talk.json \
    --input_label_h5 data/camvid/CamVid_talk.h5 \
    --input_fc_dir  data/camvid/camvid_fc \
    --input_att_dir  data/camvid/camvid_att  \
    --input_word_dir data/camvid/camvid_object_name/num  \
    --input_attr_dir data/camvid/camvid_attr_name/num    \
    --input_seg_dir  data/camvid/camvid_seg_npy/num  \
    --input_box_dir  data/camvid/camvid_box      \
    --seq_per_img 1 \
    --batch_size 32 \
    --beam_size 2 \
    --use_box   1  \
    --learning_rate 5e-4 \
    --num_layers 4 \
    --input_encoding_size 512 \
    --rnn_size 2048 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from   \
    --save_checkpoint_every 200 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 20 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3
