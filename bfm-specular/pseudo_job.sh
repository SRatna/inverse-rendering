pwd; hostname; date

# eval "$(conda shell.bash hook)"
# conda activate dl
EPOCHS=4
LR=0.05
DEVICE=cuda:0
USE_PRETRAINED=yes
VAL_SIZE=0.01
CELEBA=masked_celeba
MODEL_BASE_DIR=output-10000-20-120pc-5e-center-faces-none-2019-sfsnet-specular
DATA_DIR=/home/sadhikari/data/pseudo-data-sfsnet-bfm/$MODEL_BASE_DIR/$CELEBA/
OUT_DIR=mix-outs/$CELEBA/$MODEL_BASE_DIR
MODEL_DIR=$MODEL_BASE_DIR/sfsnet-bfm.pth

mkdir -p $OUT_DIR

python mix_sfs.py --lr $LR --out-dir $OUT_DIR --model $MODEL_DIR --data-path $DATA_DIR --epochs $EPOCHS --val-size $VAL_SIZE --use-pretrained $USE_PRETRAINED --device $DEVICE

# python ../eval_bfm.py --batch-size 5000 --path $DATA_DIR

date
# mv sfsnet_bfm_$SLURM_JOB_ID.log $DATA_DIR
cp *.py $OUT_DIR
cp job.sh $OUT_DIR