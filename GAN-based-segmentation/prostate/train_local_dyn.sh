BATCH_SIZE=1
EXP_DIR=/media/htic/NewVolume5/miccai_mri/models/localdyndisc_prostate_github/
TRAIN_PATH=/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/train_dyn_16
VALID_PATH=/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/test
sudo /home/htic/anaconda2/envs/torch4/bin/python -W ignore train_local_dyn.py --exp-dir $EXP_DIR  --batch-size $BATCH_SIZE --train-path $TRAIN_PATH --validation-path $VALID_PATH
