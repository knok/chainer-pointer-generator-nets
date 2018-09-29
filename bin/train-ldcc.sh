#!/bin/sh
# -*- coding: utf-8 -*-
#

VSIZE=8000

PREFIX=../ldcc-summarize/
WAKATI_SOURCE=$PREFIX/wakati-pp-article.txt
WAKATI_TARGET=$PREFIX/wakati-pp-abstract.txt
OUT_SOURCE=$PREFIX/all-wakati-pp-article.txt
OUT_TARGET=$PREFIX/all-wakati-pp-abstract.txt
OUT_VOCAB=$PREFIX/all-shared-vocab.txt
TRAIN_SOURCE=$PREFIX/train-wp-article.txt
TRAIN_TARGET=$PREFIX/train-wp-abstract.txt
EVAL_SOURCE=$PREFIX/eval-wp-article.txt
EVAL_TARGET=$PREFIX/eval-wp-abstract.txt
DO_PREP=0
DO_SPLIT=0
GPU=0

LOG_INTERVAL=10
VAL_INTERVAL=20
ENC_LAYERS=1
EPOCH=100
MAX_SEQ=220
BSIZE=3

get_data () {
    POPD=`pwd`
    cd ..
    if [ ! -d ldcc-summarize ] ;
    then
	git clone https://github.com/knok/ldcc-summarize
    fi
    cd ldcc-summarize && make all
    cd $POPD
}

prep () {
    if [ $DO_PREP -eq 1 ] ;
    then
	python share_preprocess.py \
	       --ignore_number --vocab-file $OUT_VOCAB \
	       --vocab-size $VSIZE \
	       $WAKATI_SOURCE $WAKATI_TARGET \
	       $OUT_SOURCE $OUT_TARGET
    fi
}

do_split () {
    if [ $DO_SPLIT -eq 1 ] ;
    then
	POPD=`pwd`
	python $PREFIX/split.py \
	       --input-article $OUT_SOURCE --input-abstract $OUT_TARGET \
	       --output-train-article $TRAIN_SOURCE \
	       --output-train-abstract $TRAIN_TARGET \
	       --output-test-article $EVAL_SOURCE \
	       --output-test-abstract $EVAL_TARGET
    fi
}

while getopts sv:ptg: opts
do
    case $opts in
	s)
	    get_data
	    ;;
	v)
	    VSIZE=$OPTARG
	    ;;
	p)
	    DO_PREP=1
	    ;;
	t)
	    DO_SPLIT=1
	    ;;
	g)
	    GPU=$OPTARG
	    ;;
    esac
done

prep
do_split

# train
python $DEBUG train.py -g $GPU \
       -e $EPOCH --max-source-sentence $MAX_SEQ \
       --validation-interval $VAL_INTERVAL -b $BSIZE \
       --log-interval $LOG_INTERVAL \
       --validation-source $EVAL_SOURCE --validation-target $EVAL_TARGET \
       $TRAIN_SOURCE $TRAIN_TARGET $OUT_VOCAB
       

exit 0
