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
mono=false  # TODO rename
splice_feats=0
cmvn=true   # cmvn, if false -> global norm

#
string=
string_command=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: generate_tf_data.sh [options] <data> <tri-model-exp-dir> <monophone-model-exp-dir> <working-dir>"
  echo " e.g.: generate_tf_data.sh --nj 35 train_20kshort_nodup exp/tri3 exp/mono output_folder"
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

data=$1
source_dir=$kaldi_path/$2
own_model_dir=$kaldi_path/$3
working_dir=$4

# we create a string depending on creating pdf or phones
if [[ "$state_based" == "true" ]]; then
    echo "---Creating labels for states---"
    string="pdf"
    string_command="pdf"
else
    echo "---Creating labels for phones---"
    string="phones"
    string_command="phones --per-frame"
fi

# create log folder in working dir
mkdir -p "$working_dir"/log $working_dir/alignments $working_dir/tf_data/train_$string \
    $working_dir/tf_data/dev_$string $working_dir/tf_data/test_$string $working_dir/features


# check if all files are existend
for f in $source_dir/tree $source_dir/final.mdl $own_model_dir/tree $own_model_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# define own model path
mdl=

# use the dev/test lattices to get the alignments for dev and test
if [ $stage -le -4 ]; then
    for dataset in dev test; do
        $cmd JOB=1:$njd $working_dir/log/lat_to_$dataset.JOB \
            lattice-1best --acoustic-scale=0.083333  "ark:gunzip -c $source_dir/decode_$dataset/lat.JOB.gz|" ark,t:$working_dir/lat_$dataset.JOB || exit 1;
        $cmd JOB=1:$njd $working_dir/log/lat_to_ali.JOB \
            nbest-to-linear ark,t:$working_dir/lat_$dataset.JOB "ark,t:|gzip -c >$working_dir/ali_$dataset.JOB.gz" || exit 1;
        rm $working_dir/lat_$dataset.* 2>/dev/null
    done
fi


if [ $stage -le -3 ]; then
    if [[ "$mono" == "true" ]]; then
        # convert triphone transitions to monophone transitions
        echo "---Convert to monophone transitions---"
        for dataset in train test dev; do
            if [[ "$dataset" == "train" ]]; then
                $cmd JOB=1:$nj $working_dir/log/get_model_alignments_${dataset}_JOB \
                    convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree "ark,t:gunzip -c $source_dir/ali.JOB.gz|" \
                    "ark,t:|gzip -c >$working_dir/ali_conv_train.JOB.gz" || exit 1;
                mdl=$own_model_dir/final.mdl
            else
                $cmd JOB=1:$njd $working_dir/log/get_model_alignments_${dataset}_JOB \
                    convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree \
                    "ark,t:gunzip -c $working_dir/ali_$dataset.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_conv_${dataset}.JOB.gz" || exit 1;
            fi
        done
    else
        echo "---Copy triphone transitions---"
        # just copy triphone transitions
        cp $source_dir/ali.*.gz $working_dir

        for dataset in train test dev; do
            if [[ "$dataset" == "train" ]]; then
                rename "s/ali/ali_conv_${dataset}/" $working_dir/ali.*
            else
                rename "s/ali/ali_conv/" $working_dir/ali_${dataset}.*
            fi

        done

        mdl=$source_dir/final.mdl

    fi
fi

if [ $stage -le -2 ]; then
    echo "---Map the transitions to $string---"
    # get phone based alignments
    for dataset in train test dev; do
        if [[ "$dataset" == "train" ]]; then
            $cmd JOB=1:$nj $working_dir/log/${string}_based_alignments_${dataset}_JOB \
                ali-to-$string_command "$mdl" "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|" \
                "ark,t:|gzip -c >$working_dir/ali_${string}_${dataset}.JOB.gz" || exit 1;
        else
            $cmd JOB=1:$njd $working_dir/log/${string}_based_alignments_${dataset}_JOB \
                ali-to-$string_command "$mdl" "ark,t:gunzip -c $working_dir/ali_conv_${dataset}.JOB.gz|" \
                "ark,t:|gzip -c >$working_dir/ali_${string}_${dataset}.JOB.gz" || exit 1;
        fi
        # create one big alignment file for train/test/dev
        gunzip -c $working_dir/ali_${string}_${dataset}.*> $working_dir/alignments/all_ali_${string}_${dataset}
    done
fi

# concat the features with the labels (alignments)
if [ $stage -le -1 ]; then
    echo "---Filter the features and merge them with the labels---"
    for dataset in train test dev; do
        mkdir -p $working_dir/tmp_${dataset}
        if [[ "$dataset" == "train" ]]; then
            python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $nj --splice $splice_feats --cmvn $cmvn $data \
                $working_dir/alignments/all_ali_${string}_${dataset} $working_dir/tmp_${dataset}
            mv $working_dir/tmp_train/features_* $working_dir/features
        else
            python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $njd --splice $splice_feats --cmvn $cmvn $dataset \
                $working_dir/alignments/all_ali_${string}_${dataset} $working_dir/tmp_${dataset}
            rm $working_dir/tmp_${dataset}/features_* 2>/dev/null
        fi
        # attention: we delete the filtered features_ without backing them up
    done
fi

# create TFRecords files for data
if [ $stage -le 0 ]; then
    echo "---Creating TFRecords files for data in tensorflow---"
    for dataset in train test dev; do
        if [[ "$dataset" == "train" ]]; then
            python $framework_path/KaldiHelper/MiscHelper.py --nj $nj --splice $splice_feats --state-based $state_based --cmvn $cmvn \
                $working_dir/stats.mat $working_dir/tmp_${dataset} $working_dir/tf_data/${dataset}_$string
        else
            python $framework_path/KaldiHelper/MiscHelper.py --nj $njd --splice $splice_feats --state-based $state_based --cmvn $cmvn \
                $working_dir/stats.mat $working_dir/tmp_${dataset} $working_dir/tf_data/${dataset}_$string
        fi
    done
fi

# Cleaning up some files which are not necessary anymore
echo "---Cleaning up---"
rm $working_dir/ali_*
rm -R $working_dir/alignments $working_dir/tmp_*

exit 0
