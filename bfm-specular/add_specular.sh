#!/bin/bash
#SBATCH --job-name=add_specular    # Job name
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --partition=htc
#SBATCH --mem=8gb                     # Job memory request
#SBATCH --time=1-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=add_specular_%j.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"
conda activate dl

MODEL_BASE_DIR=output-5k-40-80pc-10e-all-poses-none-sfsnet-2019
# MODEL_BASE_DIR=output-5k-20-100pc-more-poses-none-sfsnet
# MODEL_BASE_DIR=output-5k-20-120pc-5e-more-poses-none-sfsnet
# MODEL_BASE_DIR=output-5000-50-100pc-5e-none-15poses-2019-sfsnet
# MODEL_BASE_DIR=output-1k-100-120pc-simple-poses-none-sfsnet
# SYN_BASE_DIR=/work/ws-tmp/g051151-bfm/g051151-bfm-1640650205/g051151-bfm-1637971803/$MODEL_BASE_DIR
SYN_BASE_DIR=/work/ws-tmp/g051151-bfm/$MODEL_BASE_DIR
SPECULAR_BASE_DIR=/work/ws-tmp/g051151-bfm/$MODEL_BASE_DIR-specular
cp -r $SYN_BASE_DIR $SPECULAR_BASE_DIR
python add-specular.py --base-dir $SPECULAR_BASE_DIR

date