#!/bin/bash
#SBATCH --job-name=sfsskipnet    # Job name
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1		
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=23:00:00               # Time limit hrs:min:sec
#SBATCH --output=skipnet_%j.log   # Standard output and error log
pwd; hostname; date
eval "$(conda shell.bash hook)"
conda activate dl
EPOCHS=15
LR=0.005
DEVICE=cuda:0
USE_PRETRAINED=no
VAL_SIZE=10
ROUND=2

CELEBA=masked_celeba
SYN_DATA_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/DATA_pose_15/
OUT_DIR=skip-outs/$CELEBA/round-$ROUND
MODEL_DIR=$OUT_DIR/skipnet.pth
mkdir -p $OUT_DIR/tmp

python skip.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE

EPOCHS=10
USE_PRETRAINED=yes
# python skip.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE

# python gen_pseudo_sfsnet.py

# ./mix_job.sh
# python eval.py
# python ../eval_bfm.py --batch-size 5000 --path $DATA_DIR

date
mv skipnet_$SLURM_JOB_ID.log $OUT_DIR
cp skip.py $OUT_DIR
cp job.sh $OUT_DIR

