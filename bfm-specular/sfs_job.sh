#!/bin/bash
#SBATCH --job-name=sfs_specular    # Job name
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1		
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=23:59:59               # Time limit hrs:min:sec
#SBATCH --output=sfsnet_bfm_specular_%j.log   # Standard output and error log
pwd; hostname; date
eval "$(conda shell.bash hook)"
conda activate dl

BATCH_SIZE=60
EPOCHS=15
LR=0.05
DEVICE=cuda:0
USE_PRETRAINED=yes
VAL_SIZE=10
ROUND=4

CELEBA=masked_celeba
MODEL_BASE_DIR=output-5k-40-80pc-10e-all-poses-none-sfsnet-2019-specular
# MODEL_BASE_DIR=output-5k-20-120pc-5e-more-poses-none-sfsnet
# MODEL_BASE_DIR=output-5000-50-100pc-5e-none-15poses-2019-sfsnet
# MODEL_BASE_DIR=output-1k-100-120pc-simple-poses-none-sfsnet
SYN_DATA_DIR=/work/ws-tmp/g051151-bfm/$MODEL_BASE_DIR/
OUT_DIR=sfs-outs/$CELEBA/round-$ROUND/$MODEL_BASE_DIR
MODEL_DIR=$OUT_DIR/sfsnet-bfm.pth
mkdir -p $OUT_DIR/tmp

python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

EPOCHS=2
USE_PRETRAINED=yes
# python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

date
mv sfsnet_bfm_specular_$SLURM_JOB_ID.log $OUT_DIR
cp *.py $OUT_DIR
cp job.sh $OUT_DIR

