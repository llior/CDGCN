#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Create denominator lattices for MMI/MPE training.
# This version uses the online-nnet2 features.
#
# Creates its output in $dir/lat.*.gz

# Begin configuration section.
stage=0
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
num_threads=1
parallel_opts=  # ignored now.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./subtools/path.sh ] && . ./subtools/path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: subtools/kaldi/steps/make_denlats.sh [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
  echo "  e.g.: subtools/kaldi/steps/make_denlats.sh data/train data/lang exp/nnet2_online/nnet_a_online exp/nnet2_online/nnet_a_denlats"
  echo "Works for (delta|lda) features, and (with --transform-dir option) such features"
  echo " plus transforms."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (subtools/kaldi/utils/run.pl|subtools/kaldi/utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --sub-split <n-split>                            # e.g. 40; use this for "
  echo "                           # large databases so your jobs will be smaller and"
  echo "                           # will (individually) finish reasonably soon."
  echo "  --num-threads  <n>                # number of threads per decoding job"
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

for f in $data/wav.scp $lang/L.fst $srcdir/final.mdl $srcdir/conf/online_nnet2_decoding.conf; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

sdata=$data/split$nj

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

subtools/kaldi/utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;

oov=`cat $lang/oov.int` || exit 1;


# Compute grammar FST which corresponds to unigram decoding graph.
new_lang="$dir/"$(basename "$lang")


grep -v '^--endpoint' $srcdir/conf/online_nnet2_decoding.conf >$dir/feature.conf || exit 1;

if [ $stage -le 0 ]; then
  # mkgraph.sh expects a whole directory "lang", so put everything in one directory...
  # it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
  # final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

  cp -rH $lang $dir/

  echo "Compiling decoding graph in $dir/dengraph"
  if [ -s $dir/dengraph/HCLG.fst ] && [ $dir/dengraph/HCLG.fst -nt $srcdir/final.mdl ]; then
    echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
  else
    echo "Making unigram grammar FST in $new_lang"
    cat $data/text | subtools/kaldi/utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
      awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
      subtools/kaldi/utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > $new_lang/G.fst \
      || exit 1;
    subtools/kaldi/utils/mkgraph.sh $new_lang $srcdir $dir/dengraph || exit 1;
  fi
fi


if [ -f $data/segments ]; then
  # note: in the feature extraction, because the program online2-wav-dump-features is sensitive to the
  # previous utterances within a speaker, we do the filtering after extracting the features.
  echo "$0 [info]: segments file exists: using that."
  feats="ark,s,cs:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt ark,s,cs:- ark:- |"
else
  echo "$0 [info]: no segments file exists, using wav.scp."
  feats="ark,s,cs:online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt scp:$sdata/JOB/wav.scp ark:- |"
fi



# if this job is interrupted by the user, we want any background jobs to be
# killed too.
cleanup() {
  local pids=$(jobs -pr)
  [ -n "$pids" ] && kill $pids
}
trap "cleanup" INT QUIT TERM EXIT


if [ $sub_split -eq 1 ]; then
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode_den.JOB.log \
   nnet-latgen-faster$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
     $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
else
  # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
  # to have at most two jobs running at each time.  The idea is that if we have stragglers
  # from one job, we can be processing another one at the same time.
  rm $dir/.error 2>/dev/null

  prev_pid=
  for n in `seq $[nj+1]`; do
    if [ $n -gt $nj ]; then
      this_pid=
    elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $srcdir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
      this_pid=
    else
      sdata2=$data/split$nj/$n/split${sub_split}utt;
      split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=`echo $feats | sed "s/trans.JOB/trans.$n/g" | sed s:JOB/:$n/split${sub_split}utt/JOB/:g`

      $cmd --num-threads $num_threads JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        nnet-latgen-faster$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
        --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
          $dir/dengraph/HCLG.fst "$feats_subset" "ark:|gzip -c >$dir/lat.$n.JOB.gz" || touch $dir/.error &
      this_pid=$!
    fi
    if [ ! -z "$prev_pid" ]; then  # Wait for the previous job; merge the previous set of lattices.
      wait $prev_pid
      [ -f $dir/.error ] && echo "$0: error generating denominator lattices" && exit 1;
      rm $dir/.merge_error 2>/dev/null
      echo Merging archives for data subset $prev_n
      for k in `seq $sub_split`; do
        gunzip -c $dir/lat.$prev_n.$k.gz || touch $dir/.merge_error;
      done | gzip -c > $dir/lat.$prev_n.gz || touch $dir/.merge_error;
      [ -f $dir/.merge_error ] && echo "$0: Merging lattices for subset $prev_n failed (or maybe some other error)" && exit 1;
      rm $dir/lat.$prev_n.*.gz
      touch $dir/.done.$prev_n
    fi
    prev_n=$n
    prev_pid=$this_pid
  done
fi


echo "$0: done generating denominator lattices."
