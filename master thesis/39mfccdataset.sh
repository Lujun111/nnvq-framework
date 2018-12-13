#!/bin/bash
#
# yezi@25.09.2018
# preprocess the dataset for the neural network in tensorflow/pytorch
#

set -e -o pipefail -u

. ./cmd.sh
. ./path.sh

stage=0
nj=35
cmd=utils/run.pl

. ./utils/parse_options.sh # accept options

#if [ $# != 3 ]; then
#    echo "format error: $0 [options]  <feats-dir> <ali-dir> <mono-dir> "
#    exit 1;
#fi

#feats_dir=$1
#ali_dir=$2
#mono_dir=$3

#mkdir -p $ali_dir/dataset

#echo " 13mfcc --> apply-cmvn --> delta+deltadelta "
#run.pl JOB=1:$nj $feats_dir/log_self/13_cmvn_39.JOB.log \
#    apply-cmvn --norm-vars=true --utt2spk=ark:$feats_dir/JOB/utt2spk scp:$feats_dir/JOB/cmvn.scp \
#	scp:$feats_dir/JOB/feats.scp ark:- \| add-deltas --delta-order=2 ark:- \
#	ark,scp:$feats_dir/JOB/39mfcc.ark,$feats_dir/JOB/39mfcc.scp

#echo " convert ali of tri3 to mono "
#run.pl JOB=1:$nj $ali_dir/dataset/log_self/tri3_to_mono.JOB.log \
#    gunzip -c $ali_dir/ali.JOB.gz \| convert-ali $ali_dir/final.mdl $mono_dir/final.mdl $mono_dir/tree \
#	ark:- ark:- \|  ali-to-pdf $mono_dir/final.mdl ark:- \
#	"ark,scp:$ali_dir/dataset/127pdf.JOB.ark,$ali_dir/dataset/127pdf.JOB.scp"	 




#ali_dir=/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3
#echo " convert ali of pdf of tri3 model "
#run.pl JOB=1:$nj $ali_dir/4152dataset/log_self/tri3_to_pdf.JOB.log \
#    gunzip -c $ali_dir/ali.JOB.gz \| ali-to-pdf $ali_dir/final.mdl ark:- \
#	"ark,scp:$ali_dir/4152dataset/4152pdf.JOB.ark,$ali_dir/4152dataset/4152pdf.JOB.scp"

echo " get  the pdf counts file "
ali_dir=/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3
num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
echo $num_pdf
gunzip -c $ali_dir/ali*.gz | ali-to-pdf $ali_dir/final.mdl ark:- ark:- | analyze-counts --verbose=1 --binary=false --counts-dim=4152 ark:- $ali_dir/4152_pdf.counts




































