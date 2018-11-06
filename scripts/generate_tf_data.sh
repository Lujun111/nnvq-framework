#!/bin/bash

# cd to kaldi path and add path.sh and run.sh
kaldi_path=~/kaldi/egs/tedlium/s5_r2
framework_path=~/nnvq-framework
cd $kaldi_path
. ./cmd.sh
. ./path.sh
# cd back to project folder
# cd ~/nnvq-framework
source ~/tensorflow_py3/bin/activate

nj=20
cmd=run.pl
stage=-4
state_based=true
train_data=true


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: generate_tf_data.sh [options] <data> <alignment-dir> <own-exp-dir> <working-dir>"
  echo " e.g.: generate_tf_data.sh --nj 35 train_20kshort_nodup exp/tri3 exp/mono output_folder"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --state-based <true/false>                       # state-base or phoneme labels"
  echo "  --train_data <true/false>                        # create only train data or also dev/test data"
  exit 1;
fi

data=$1
source_dir=$kaldi_path/$2
own_model_dir=$kaldi_path/$3
working_dir=$framework_path/scripts/$4

# create log folder in working dir
mkdir -p $working_dir/log $working_dir/alignments $working_dir/tf_data


# check if all files are existend
for f in $source_dir/tree $source_dir/final.mdl $own_model_dir/tree $own_model_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# define own model path
mdl=$own_model_dir/final.mdl


# convert alignments to own model
echo "---convert alignment to own model---"
if [ $stage -le -3 ]; then
    # get all alignments for own model
    $cmd JOB=1:$nj $working_dir/log/get_model_alignments_JOB convert-ali $source_dir/final.mdl $own_model_dir/final.mdl $own_model_dir/tree\
        "ark,t:gunzip -c $source_dir/ali.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_conv.JOB.gz"

fi

ali_dir=    # setting ali_dir var
# convert alignments to pdf based or phoneme based
if [ $stage -le -2 ]; then
    if [[ "$state_based" == "true" ]]; then
        echo "---map the alignments to state-based labels---"
        # get state based alignments
        $cmd JOB=1:35 $working_dir/log/state_based_alignments_JOB ali-to-pdf $mdl "ark,t:gunzip -c $working_dir/ali_conv.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_pdf.JOB.gz"

        # remove old files
        rm $working_dir/ali_conv.* 2>/dev/null

        # create one big alignment file
        gunzip -c $working_dir/ali_pdf.*> $working_dir/alignments/all_ali_pdf
        ali_dir=$working_dir/alignments/all_ali_pdf
        # remove old files
        rm $working_dir/ali_pdf.* 2>/dev/null
    else
        echo "---map the alignments to phoneme-based labels---"
        # get phoneme based alignments
        $cmd JOB=1:35 $working_dir/log/phoneme_based_alignments_JOB ali-to-phones --per-frame $mdl "ark,t:gunzip -c $working_dir/ali_conv.JOB.gz|" "ark,t:|gzip -c >$working_dir/ali_pho.JOB.gz"

        # remove old files
        rm $working_dir/ali_conv.* 2>/dev/null

        # create one big alignment file
        gunzip -c $working_dir/ali_pho.*> $working_dir/alignments/all_ali_phoneme
        ali_dir=$working_dir/alignments/all_ali_phoneme

        # remove old files
        rm $working_dir/ali_pho.* 2>/dev/null

    fi
fi

# concat the features with the labels (alignments)
# we also create the features for training because we don't have alignments for
# all the training data
if [ $stage -le -1 ]; then
    echo "---filter the features and merge them with the labels ---"
    mkdir -p $working_dir/tmp
    python $framework_path/KaldiHelper/KaldiMiscHelper.py --nj $nj $data $ali_dir $working_dir/tmp
    # # remove some files
    rm $working_dir/tmp/features_{1..35} 2>/dev/null
fi

if [ $stage -le 0 ]; then
    echo "---creating TFRecords files for training in tensorflow---"
    python $framework_path/KaldiHelper/MiscHelper.py --nj $nj --state-based $state_based $framework_path/stats.mat $working_dir/tmp $working_dir/tf_data
fi

exit 0
