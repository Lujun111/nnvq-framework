#!/bin/bash
#
# yezi@18.09.2018
# decoding for tdnn in tensroflow
#

set -e -o pipefail -u

. ./cmd.sh
. ./path.sh


## Begin configuration section
stage=0
nj=30
cmd=utils/run.pl

acwt=0.10
beam=13.0
lattice_beam=8.0
min_active=200
max_active=7000
max_mem=50000000

skip_scoring=false
scoring_opts="--min-lmwt 4 --max-lmwt 15"

. ./utils/parse_options.sh || exit 1; # accept options

if [ $# != 5 ]; then
   echo "format error: $0 [options] <data-dir> <gmm-model.mdl> <graph-dir> <decode-dir> <postark>"
   exit 1;
fi


data_dir=$1
gmm_mdl=$2
graph_dir=$3
decode_dir=$4
postark=$5

mkdir -p $decode_dir/log

echo "begin to check the expected files"

for x in $gmm_mdl $graph_dir/HCLG.fst $graph_dir/words.txt; do
   [ ! -f $x ] && echo "$0: no such file $x " && exit 1;
done

echo "begin to decode to generate lattice file"

if [ $stage -le 1 ]; then
   $cmd JOB=1:$nj $decode_dir/log/decode.JOB.log \
	latgen-faster-mapped --min-active=$min_active --max-active=$max_active --max-mem=$max_mem \
	--beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
	--word-symbol-table=$graph_dir/words.txt $gmm_mdl $graph_dir/HCLG.fst "$postark" \
	"ark:|gzip -c > $decode_dir/lat.JOB.gz" || exit 1;
fi

echo "begin to score to get WER"

if [ $stage -le 2 ]; then
   ./local/score.sh $scoring_opts --cmd "$cmd" $data_dir $graph_dir $decode_dir
fi

echo "get the best WER"

for d in $decode_dir; do grep Sum $d/score*/*ys | utils/best_wer.sh; done

echo "Decoding and scoring done"
exit 0
