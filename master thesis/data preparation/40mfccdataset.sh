#!/bin/bash
#
# yezi@25.09.2018
# preprocess the dataset for the neural network in tensorflow/pytorch
#

set -e -o pipefail -u

. ./cmd.sh
. ./path.sh

nj=30 #normally it depends on how many ali.*.gz file
cmd=utils/run.pl

. ./utils/parse_options.sh || exit 1; # accept options

if [ $# != 2 ]; then
    echo "format error: $0 [options] <ali-dir> <feats-dir> "
    exit 1;
fi

ali_dir=$1
feats_dir=$2

mkdir -p $ali_dir/dataset

echo "firstly use the kaldi command 'ali-to-pdf' to convert all of the ali.*.gz to scp format with pdfs "
$cmd JOB=1:$nj $ali_dir/log_self/ali_to_pdf.JOB.log \
	gunzip -c $ali_dir/ali.JOB.gz \| ali-to-pdf $ali_dir/final.mdl ark:- \
	"ark,scp:$ali_dir/dataset/pdf.JOB.ark,$ali_dir/dataset/pdf.JOB.scp"

echo "next we need to use kaldi command apply-cmvn to the every feats.scp"
##this part code can be also used for test data##
$cmd JOB=1:$nj $feats_dir/log_self/apply_cmvn.JOB.log \
	apply-cmvn --norm-vars=true --utt2spk=ark:$feats_dir/JOB/utt2spk scp:$feats_dir/JOB/cmvn.scp \
	scp:$feats_dir/JOB/feats.scp ark,scp:$feats_dir/JOB/feats_cmvn.ark,$feats_dir/JOB/feats_cmvn.scp

echo "then cat the feats.scp to generate the train and validation dataset"
cat $feats_dir/1/feats_cmvn.scp $feats_dir/2/feats_cmvn.scp $feats_dir/3/feats_cmvn.scp \
	$feats_dir/4/feats_cmvn.scp $feats_dir/5/feats_cmvn.scp $feats_dir/6/feats_cmvn.scp \
	$feats_dir/7/feats_cmvn.scp $feats_dir/8/feats_cmvn.scp $feats_dir/9/feats_cmvn.scp \
	$feats_dir/10/feats_cmvn.scp $feats_dir/11/feats_cmvn.scp $feats_dir/12/feats_cmvn.scp \
	$feats_dir/13/feats_cmvn.scp $feats_dir/14/feats_cmvn.scp $feats_dir/15/feats_cmvn.scp \
	$feats_dir/16/feats_cmvn.scp $feats_dir/17/feats_cmvn.scp $feats_dir/18/feats_cmvn.scp \
	$feats_dir/19/feats_cmvn.scp $feats_dir/20/feats_cmvn.scp $feats_dir/21/feats_cmvn.scp \
	$feats_dir/22/feats_cmvn.scp $feats_dir/23/feats_cmvn.scp $feats_dir/24/feats_cmvn.scp \
	$feats_dir/25/feats_cmvn.scp > $feats_dir/feats_train.scp


cat $feats_dir/26/feats_cmvn.scp $feats_dir/27/feats_cmvn.scp $feats_dir/28/feats_cmvn.scp \
	$feats_dir/29/feats_cmvn.scp $feats_dir/30/feats_cmvn.scp > $feats_dir/feats_dev.scp

echo "finally cat the pdf.*.scp to generate the train and validation dataset"
cat $ali_dir/dataset/pdf.1.scp $ali_dir/dataset/pdf.2.scp $ali_dir/dataset/pdf.3.scp \
	$ali_dir/dataset/pdf.4.scp $ali_dir/dataset/pdf.5.scp $ali_dir/dataset/pdf.6.scp \
	$ali_dir/dataset/pdf.7.scp $ali_dir/dataset/pdf.8.scp $ali_dir/dataset/pdf.9.scp \
	$ali_dir/dataset/pdf.10.scp $ali_dir/dataset/pdf.11.scp $ali_dir/dataset/pdf.12.scp \
	$ali_dir/dataset/pdf.13.scp $ali_dir/dataset/pdf.14.scp $ali_dir/dataset/pdf.15.scp \
	$ali_dir/dataset/pdf.16.scp $ali_dir/dataset/pdf.17.scp $ali_dir/dataset/pdf.18.scp \
	$ali_dir/dataset/pdf.19.scp $ali_dir/dataset/pdf.20.scp $ali_dir/dataset/pdf.21.scp \
	$ali_dir/dataset/pdf.22.scp $ali_dir/dataset/pdf.23.scp $ali_dir/dataset/pdf.24.scp \
	$ali_dir/dataset/pdf.25.scp > $ali_dir/dataset/pdf_train.scp


cat $ali_dir/dataset/pdf.26.scp $ali_dir/dataset/pdf.27.scp $ali_dir/dataset/pdf.28.scp \
	$ali_dir/dataset/pdf.29.scp $ali_dir/dataset/pdf.30.scp > $ali_dir/dataset/pdf_dev.scp





