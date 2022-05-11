#!/bin/bash
#SBATCH --job-name=sfs    # Job name
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1		
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=23:59:59               # Time limit hrs:min:sec
#SBATCH --output=sfsnet_%j.log   # Standard output and error log
pwd; hostname; date
eval "$(conda shell.bash hook)"
conda activate dl

BATCH_SIZE=60
EPOCHS=6
LR=0.05
DEVICE=cuda:0
USE_PRETRAINED=no
VAL_SIZE=0.50
ROUND=1

CELEBA=masked_celeba
SYN_DATA_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/DATA_pose_15/
OUT_DIR=sfs-outs/$CELEBA/round-$ROUND
MODEL_DIR=$OUT_DIR/sfsnet.pth
mkdir -p $OUT_DIR/tmp

python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

# EPOCHS=3
# USE_PRETRAINED=yes
# python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

# setting parameters for pseudo data-generation using SkipNet
SKIP_ROUND=1
SKIP_OUT_DIR=skip-outs/$CELEBA/round-$SKIP_ROUND/$MODEL_BASE_DIR
SKIP_MODEL_DIR=$SKIP_OUT_DIR/skipnet.pth
PSEUDO_DATA_DIR=/work/ws-tmp/g051151-bfm/pseudo-data-skipnet/$MODEL_BASE_DIR/$CELEBA/round-$SKIP_ROUND/
CELEBA_DATA_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/$CELEBA/

python gen_pseudo_skipnet.py --model $SKIP_MODEL_DIR --celeba-data-dir $CELEBA_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --device $DEVICE --batch-size $BATCH_SIZE

MIX_OUT_DIR=mix-outs/$CELEBA/round-$ROUND
mkdir -p $MIX_OUT_DIR/tmp
EPOCHS=4
VAL_SIZE=0.50

python mix_sfs.py --lr $LR --out-dir $MIX_OUT_DIR --model $MODEL_DIR --syn-data-dir $SYN_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE  --batch-size $BATCH_SIZE

mkdir -p eval-outs/

PHOTOFACE_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/Photoface_processed/
BATCH=12000 # total size of photoface (can be less too)
python eval.py --device $DEVICE --out-dir $OUT_DIR --mix-out-dir $MIX_OUT_DIR --round $ROUND --data-dir $PHOTOFACE_DIR --batch $BATCH

date
mv sfsnet_$SLURM_JOB_ID.log $MIX_OUT_DIR
cp *.py $MIX_OUT_DIR
cp job.sh $MIX_OUT_DIR

