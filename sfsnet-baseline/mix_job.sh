pwd; hostname; date

# eval "$(conda shell.bash hook)"
# conda activate dl
EPOCHS=2
LR=0.05
DEVICE=cuda:0
USE_PRETRAINED=yes
VAL_SIZE=0.01
ROUND=1

CELEBA=masked_celeba
MODEL_BASE_DIR=output-10000-20-120pc-5e-center-faces-none-2019-sfsnet-specular
PSEUDO_DATA_DIR=/home/sadhikari/data/pseudo-data-sfsnet-bfm/$MODEL_BASE_DIR/$CELEBA/
SYN_DATA_DIR=../bfm-sfsnet-data/$MODEL_BASE_DIR/
OUT_DIR=mix-outs/$CELEBA/round-$ROUND/$MODEL_BASE_DIR
SFS_OUT_DIR=sfs-outs/$CELEBA/round-$ROUND/$MODEL_BASE_DIR
MODEL_DIR=$SFS_OUT_DIR/sfsnet-bfm.pth
mkdir -p $OUT_DIR/tmp

python mix_sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --syn-data-dir $SYN_DATA_DIR --pseudo-data-dir $PSEUDO_DATA_DIR --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE

# python eval.py
# python ../eval_bfm.py --batch-size 5000 --path $DATA_DIR

date
# mv sfsnet_bfm_$SLURM_JOB_ID.log $DATA_DIR
cp *.py $OUT_DIR
cp job.sh $OUT_DIR

