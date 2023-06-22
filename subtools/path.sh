KALDI_ROOT=/work/kaldi/tools
if [ -d $KALDI_ROOT ];then
    export KALDI_ROOT=/work/kaldi/
    [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
    export PATH=$PWD/subtools/kaldi/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
    [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
    . $KALDI_ROOT/tools/config/common_path.sh
    export LC_ALL=C
    PATH=$PATH:/work/kaldi/src/cudafeatbin
    export PATH
fi

KALDI_ROOT=/data/kaldi/tools
if [ -d $KALDI_ROOT ];then
    export KALDI_ROOT=/data/kaldi/  #`pwd`/../../..
    [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
    export PATH=$PWD/subtools/kaldi/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
    [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
    . $KALDI_ROOT/tools/config/common_path.sh
    export LC_ALL=C
    PATH=$PATH:/data/kaldi/src/cudafeatbin
    export PATH
fi

