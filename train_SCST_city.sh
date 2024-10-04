id="transformer"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train.py --id transformer  \
    --caption_model transformer \
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
    --input_json data/cityscapes/cityscapes_talk.json \
    --input_label_h5 data/cityscapes/cityscapes_talk.h5 \
    --input_fc_dir  data/cityscapes/cityscapes_fc \
    --input_att_dir  data/cityscapes/cityscapes_att  \
    --input_word_dir data/cityscapes/cityscapes_object_name/num  \
    --input_attr_dir data/cityscapes/cityscapes_attr_name/num    \
    --input_box_dir  data/cityscapes/cityscapes_box      \
    --seq_per_img 5 \
    --batch_size 16 \
    --beam_size 2 \
    --use_box   1  \
    --learning_rate 5e-4 \
    --num_layers 6 \
    --input_encoding_size 512 \
    --att_hid_size 512  \
    --rnn_size 512 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from     \
    --save_checkpoint_every 40 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 10 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3

python train.py --id $id \
    --caption_model transformer  \
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
    --input_json data/cityscapes/cityscapes_talk.json \
    --input_label_h5 data/cityscapes/cityscapes_talk.h5 \
    --input_fc_dir  data/cityscapes/cityscapes_fc \
    --input_att_dir  data/cityscapes/cityscapes_att  \
    --input_word_dir   data/cityscapes/cityscapes_object_name/num  \
    --input_attr_dir    data/cityscapes/cityscapes_attr_name/num  \
    --input_box_dir  data/cityscapes/cityscapes_box \
    --seq_per_img 5 \
    --batch_size 8 \
    --beam_size 2 \
    --num_layers 6 \
    --input_encoding_size 512 \
    --rnn_size 512 \
    --language_eval 1 \
    --val_images_use -1 \
    --save_checkpoint_every 60 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 5e-5  \
    --max_epochs 20 \
    --self_critical_after 0 \
    --cached_tokens cityscapes-idxs   \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau 
