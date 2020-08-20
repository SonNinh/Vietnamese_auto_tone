python train2.py \
-data_pkl=/mnt/data/sonninh/vietnamese_tone/pre_processed/mini_vietnamese.pkl \
-epoch=50 \
-batch_size=300 \
-save_dir=/mnt/data/sonninh/trained_models/vietnamese_tone_enc_only \

# -FT \
# -resume_from=/mnt/data/sonninh/trained_models/vietnamese_tone_no_teacher/best_none_teacher.chkpt \

-emb_dim=512 \
-feedforward_dim=1024 \
-max_seq_len=64 \
-dropout=0.1 \
-n_block=4 \
-attn_dim=64 \
-n_head=8 \
-lr=0.0001