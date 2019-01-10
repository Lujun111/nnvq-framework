#!/bin/bash

framework_path=~/nnvq-framework

# check if all imports in the python are importable and change to your python path
source ~/tensorflow_py3/bin/activate

nj=20   # number of jobs for alignments
njd=30  # number of jobs for decoding
cmd=run.pl
stage=0
state_based=true
mono=false  # TODO rename
splice_feats=0
cmvn=true   # cmvn, if false -> global norm
dim=39
dropout=0.1

. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: train_experiments.sh [options] <model_save_folder>"
  echo " e.g.: train_experiments.sh model_checkpoint"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs for train data"
  echo "  --njd <njd>                                      # number of parallel jobs for dev/test data"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --state-based <true/false>                       # state-base or phoneme labels"
  echo "  --mono <true/false>                              # create monophone labels or triphone labels"
  echo "  --splice-feats <num>                             # splice feats with context, dafault single frame"
  echo "  --cmvn <true/false>                              # use cmvn or global normalization"
  exit 1;
fi

model_checkpoint=$1
tensorboard=${model_checkpoint}/tensorboard

# create folders
mkdir -p ${tensorboard}


if [ $stage -le 0 ]; then
    # start train_experiments
    for i in 1 3 5; do
        # create path and folder
        current_path=$model_checkpoint/nnvq_all_${i}f
        mkdir -p $current_path

        # define data paths
        path_train="tf_data/splice_${i}f/train_pdf_all_splice_${i}f_cmn"
        path_test="tf_data/splice_${i}f/test_pdf_all_splice_${i}f_cmn"
        path_dev="tf_data/splice_${i}f/dev_pdf_all_splice_${i}f_cmn"

        # train nnvq
        cd ..
        python main.py --cb_size 1000 --tensorboard scripts/$tensorboard --checkpoint scripts/$current_path \
            --path_train $path_train --path_test $path_test --path_dev $path_dev --dim_features $((${i}*$dim)) \
            --dropout $dropout
        cd -
    done
fi

if [ $stage -le 1 ]; then
    # start train_experiments for mono
    f=1 # == splice
    for i in 400 700 1000 1500; do
        # create path and folder
        current_path=$model_checkpoint/nnvq_20k_${i}_${f}f
        mkdir -p $current_path

        # define data paths
        path_train="tf_data/splice_${f}f/train_pdf_20k_splice_${f}f_cmn"
        path_test="tf_data/splice_${f}f/test_pdf_20k_splice_${f}f_cmn"
        path_dev="tf_data/splice_${f}f/dev_pdf_20k_splice_${f}f_cmn"

        # train nnvq
        cd ..
        python main.py --cb_size $i --tensorboard scripts/$tensorboard --checkpoint scripts/$current_path \
            --path_train $path_train --path_test $path_test --path_dev $path_dev --dim_features $((${f}*$dim)) \
            --dropout $dropout
        cd -
    done
fi

exit 0
