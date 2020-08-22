python train2.py \
-data_pkl=/mnt/data/sonninh/vietnamese_tone/pre_processed/vietnamese.pkl \
-epoch=10 \
-batch_size=600 \
-save_dir=/mnt/data/sonninh/trained_models/vietnamese_tone_enc_bigdata \
-emb_dim=256 \
-feedforward_dim=1024 \
-max_seq_len=64 \
-dropout=0.1 \
-n_block=6 \
-attn_dim=64 \
-n_head=8 \
-lr=0.00008 \
# -FT \
# -resume_from=/mnt/data/sonninh/trained_models/vietnamese_tone_enc_only/best.chkpt \
