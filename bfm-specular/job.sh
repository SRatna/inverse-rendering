#!/bin/bash
#SBATCH --job-name=specular    # Job name
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
USE_PRETRAINED=no
VAL_SIZE=10
ROUND=4

CELEBA=masked_celeba
# note round 1 uses reconstruction loss too during sfs while round 4 did not but round 4 is better although shading is worse
MODEL_BASE_DIR=output-5k-40-80pc-10e-all-poses-none-sfsnet-2019-specular
# MODEL_BASE_DIR=output-5k-20-120pc-5e-more-poses-none-sfsnet
# MODEL_BASE_DIR=output-5000-50-100pc-5e-none-15poses-2019-sfsnet
# MODEL_BASE_DIR=output-1k-100-120pc-simple-poses-none-sfsnet
SYN_DATA_DIR=/work/ws-tmp/g051151-bfm/$MODEL_BASE_DIR/
OUT_DIR=sfs-outs/$CELEBA/round-$ROUND/$MODEL_BASE_DIR
MODEL_DIR=$OUT_DIR/sfsnet-bfm.pth
mkdir -p $OUT_DIR/tmp

# python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

EPOCHS=2
USE_PRETRAINED=yes
# python sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-dir $SYN_DATA_DIR  --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE --batch-size $BATCH_SIZE

# ROUND=10

PSEUDO_DATA_DIR=/work/ws-tmp/g051151-bfm/pseudo-data-sfsnet-bfm-specular/$MODEL_BASE_DIR/$CELEBA/round-$ROUND/
CELEBA_DATA_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/$CELEBA/

# python gen_pseudo_sfsnet.py --model $MODEL_DIR --celeba-data-dir $CELEBA_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --device $DEVICE --batch-size $BATCH_SIZE

# setting for skip net data gen
SKIP_ROUND=1
SKIP_OUT_DIR=skip-outs/$CELEBA/round-$SKIP_ROUND/$MODEL_BASE_DIR
SKIP_MODEL_DIR=$SKIP_OUT_DIR/skipnet-bfm.pth
PSEUDO_DATA_DIR=/work/ws-tmp/g051151-bfm/pseudo-data-skipnet-bfm-specular/$MODEL_BASE_DIR/$CELEBA/round-$SKIP_ROUND/

# python gen_pseudo_skipnet.py --model $SKIP_MODEL_DIR --celeba-data-dir $CELEBA_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --device $DEVICE --batch-size $BATCH_SIZE

ROUND=4
MIX_OUT_DIR=mix-skip-outs/$CELEBA/round-$ROUND/$MODEL_BASE_DIR
mkdir -p $MIX_OUT_DIR/tmp
EPOCHS=6
VAL_SIZE=10

# python mix_sfs.py --lr $LR --out-dir $MIX_OUT_DIR --model $MODEL_DIR --syn-data-dir $SYN_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE  --batch-size $BATCH_SIZE

PHOTOFACE_DIR=/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1640909401/g051151-sfsnet-1633129803/Photoface_processed/
BATCH=12000 # total size of photoface (can be less too)
python eval.py --device $DEVICE --out-dir $OUT_DIR --mix-out-dir $MIX_OUT_DIR --round $ROUND --data-dir $PHOTOFACE_DIR --batch $BATCH

date
mv sfsnet_bfm_specular_$SLURM_JOB_ID.log $MIX_OUT_DIR
cp *.py $MIX_OUT_DIR
cp job.sh $MIX_OUT_DIR

