#!/bin/bash

# cd to kaldi path and add path.sh and run.sh
kaldi_path=~/kaldi/egs/tedlium/s5_r2
framework_path=~/nnvq-framework
cd $kaldi_path
. ./cmd.sh
. ./path.sh
cd -
# cd back to project folder
# cd ~/nnvq-framework

# check if all imports in the python are importable and change to your python path
source ~/tensorflow_py3/bin/activate

nj=20   # number of jobs for alignments
njd=30  # number of jobs for decoding
cmd=run.pl
stage=-5
state_based=true
train_data=true
mono=true

#
string=


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: generate_tf_data.sh [options] <data> <tri-model-exp-dir> <monophone-model-exp-dir> <working-dir>"
  echo " e.g.: generate_tf_data.sh --nj 35 train_20kshort_nodup exp/tri3 exp/mono output_folder"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --state-based <true/false>                       # state-base or phoneme labels"
  echo "  --mono <true/false>                              # create monophone labels or triphone labels"
  echo "  --train_data <true/false>                        # create only train data or also dev/test data"
  exit 1;
fi

data=$1
source_dir=$kaldi_path/$2
own_model_dir=$kaldi_path/$3
working_dir=$4

# create log folder in working dir
mkdir -p $working_dir/log $working_dir/alignments $working_dir/tf_data/train $working_dir/tf_data/dev $working_dir/tf_data/test


# check if all files are existend
for f in $source_dir/tree $source_dir/final.mdl $own_model_dir/tree $own_model_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [[ "$state_based" == "true" ]]; then
    string="-pdf"
else
    string="-phones"
fi

# define own model path
mdl=


if [ $stage -le -4 ]; then
    # use the dev/test lattices to get the alignments for dev and test
    for dataset in dev test; do
        $cmd JOB=1:$njd $working_dir/log/lat_to_$dataset.JOB lattice-1best --acoustic-scale=0.083333  "ark:gunzip -c $source_dir/decode_$dataset/lat.JOB.gz|" ark,t:$working_dir/lat_$dataset.JOB
        $cmd JOB=1:$njd $working_dir/log/lat_to_ali.JOB nbest-to-linear ark,t:$working_dir/lat_$dataset.JOB "ark,t:|gzip -c >$working_dir/ali_$dataset.JOB.gz"
        rm $working_dir/lat_$dataset.* 2>/dev/null
    done
fi


if [ $stage -le -3 ]; then
# convert alignments to own model
    if [[ "$mono" == "true" ]]; then
        echo "---convert alignments to monophone model---"
        for dataset in train test dev; do
            if [[ "$dataset" == "train" ]]; then
                $cmd JOB=1:$nj $working_dir/log/get_model_alignments_${dataset}_JOB convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree\
                    "ark,t:gunzip -c $source_dir/ali.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_conv_train.JOB.gz"
                mdl=$own_model_dir/final.mdl
            else
                $cmd JOB=1:$njd $working_dir/log/get_model_alignments_${dataset}_JOB convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree\
                    "ark,t:gunzip -c $working_dir/ali_$dataset.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_conv_${dataset}.JOB.gz"
            fi
        done

        # get all alignments for monophone model
        # $cmd JOB=1:$nj $working_dir/log/get_model_alignments_JOB convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree\
        #     "ark,t:gunzip -c $source_dir/ali.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_train.JOB.gz"
        # mdl=$own_model_dir/final.mdl
    else
        echo "---copy alignments of triphone model---"
        # else copy alignments and rename them from triphone model
        cp $source_dir/ali.*.gz $working_dir
        rename 's/ali/ali_train/' $working_dir/ali.*
        mdl=$source_dir/final.mdl

    fi
fi


# ali_dir=    # setting ali_dir var
# # convert alignments to pdf based or phoneme based
# if [ $stage -le -2 ]; then
#     if [[ "$state_based" == "true" ]]; then
#         echo "---map the alignments to $string---"
#         # get state based alignments
#         for dataset in train test dev; do
#             if [[ "$dataset" == "train" ]]; then
#                 $cmd JOB=1:$nj $working_dir/log/state_based_alignments_${dataset}_JOB ali-to-pdf $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
#                     "ark,t:|gzip -c >$working_dir/ali_pdf_${dataset}.JOB.gz"
#             else
#                 $cmd JOB=1:$njd $working_dir/log/state_based_alignments_${dataset}_JOB ali-to-pdf $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
#                     "ark,t:|gzip -c >$working_dir/ali_pdf_${dataset}.JOB.gz"
#             fi
#             gunzip -c $working_dir/ali_pdf_${dataset}.*> $working_dir/alignments/all_ali_pdf_${dataset}
#             # ali_dir_${dataset}="$working_dir/alignments/all_ali_pdf_${dataset}"
#             # remove old files
#             # rm $working_dir/ali_* 2>/dev/null
#         done
#
#         # # create one big alignment file
#         # gunzip -c $working_dir/ali_pdf.*> $working_dir/alignments/all_ali_pdf
#         # ali_dir=$working_dir/alignments/all_ali_pdf
#         # # remove old files
#         # rm $working_dir/ali_pdf.* 2>/dev/null
#     else
#         echo "---map the alignments to $string---"
#         # get phoneme based alignments
#         for dataset in train test dev; do
#             if [[ "$dataset" == "train" ]]; then
#                 $cmd JOB=1:$nj $working_dir/log/phoneme_based_alignments_${dataset}_JOB ali-to-phones --per-frame $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
#                     "ark,t:|gzip -c >$working_dir/ali_pho_${dataset}.JOB.gz"
#             else
#                 $cmd JOB=1:$njd $working_dir/log/phoneme_based_alignments_${dataset}_JOB ali-to-phones --per-frame $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
#                     "ark,t:|gzip -c >$working_dir/ali_pho_${dataset}.JOB.gz"
#             fi
#             # create one big alignment file
#             gunzip -c $working_dir/ali_pho_${dataset}.*> $working_dir/alignments/all_ali_phoneme_${dataset}
#         done
#
#         # remove old files
#         # rm $working_dir/ali_* 2>/dev/null
#     fi
#
#     # clean up files
#     rm $working_dir/ali_* 2>/dev/null
# fi
if [ $stage -le -2 ]; then
    echo "---map the alignments to $string---"
    # get phoneme based alignments
    for dataset in train test dev; do
        if [[ "$dataset" == "train" ]]; then
            echo "here"
            $cmd --num-threads 4 JOB=1:$nj $working_dir/log/${string}_based_alignments_${dataset}_JOB\
                ali-to$string --per-frame $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
                    "ark,t:|gzip -c >$working_dir/ali_${string}_${dataset}.JOB.gz"
        else
            $cmd --num-threads 4 JOB=1:$njd $working_dir/log/${string}_based_alignments_${dataset}_JOB\
                ali-to$string --per-frame $mdl "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|"\
                    "ark,t:|gzip -c >$working_dir/ali_${string}_${dataset}.JOB.gz"
        fi
        # create one big alignment file
        gunzip -c $working_dir/ali_${string}_${dataset}.*> $working_dir/alignments/all_ali_${string}_${dataset}
    done

    # rm $working_dir/ali_* 2>/dev/null
fi
exit 1;

# concat the features with the labels (alignments)
# we also create the features for training because we don't have alignments for
# all the training data
if [ $stage -le -1 ]; then
    echo "---filter the features and merge them with the labels ---"
    for dataset in train test dev; do
        $working_dir/tmp_${dataset}
        if [[ "$dataset" == "train" ]]; then
            python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $nj $data $working_dir/alignments/all_ali_pdf_${dataset}r $working_dir/tmp_${dataset}
        else
            python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $njd $data $working_dir/alignments/all_ali_pdf_${dataset}r $working_dir/tmp_${dataset}
        fi
    done
    python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $nj $data $ali_dir $working_dir/tmp
    # # remove some files
    rm $working_dir/tmp/features_* 2>/dev/null
fi

if [ $stage -le 0 ]; then
    echo "---creating TFRecords files for training in tensorflow---"
    python $framework_path/KaldiHelper/MiscHelper.py --nj $nj --state-based $state_based $framework_path/stats.mat $working_dir/tmp $working_dir/tf_data/train
fi

exit 0
