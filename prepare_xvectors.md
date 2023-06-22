#Copyright xmuspeech (Author: wangjie 2022-11-17)

## [1] prepare data 

```shell
voxceleb2_path=/data/zhd_data/Multimodal/voxceleb2
dihard_2020_dev_path=/tsdata/diarization/DIHARDIII/LDC2020E12_Third_DIHARD_Challenge_Development_Data
dihard_2020_eval_path=/tsdata/diarization/DIHARDIII/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete
prefix=fbank_81
feat_type=fbank
feat_conf=subtools/conf/sre-fbank-81.conf

subtools/recipe/voxceleb/prepare/make_voxceleb2.pl $voxceleb2_path dev data/$prefix/voxceleb2_train
local/make_dihard_2018_dev.sh $dihard_2020_dev_path data/$prefix/dihard_2020_dev
local/make_dihard_2018_dev.sh $dihard_2020_eval_path data/$prefix/dihard_2020_eval

subtools/makeFeatures.sh --nj 30 data/$prefix/voxceleb2_train/ $feat_type $feat_conf
subtools/makeFeatures.sh --nj 30 data/$prefix/dihard_2020_dev/ $feat_type $feat_conf
subtools/makeFeatures.sh --nj 30 data/$prefix/dihard_2020_eval/ $feat_type $feat_conf
```

## [2] extract xvector

```shell
#[3]extractor xvector for voxceleb1 and voxceleb2
subtools/runPytorchLauncher.sh subtools/recipe/voxcelebSRC_lgcn/run-resnet34-fbank-81-benchmark.py --stage=4

#[5]extract xvector of sliding windows for DIHARDIII
nnet_dir=exp/vox2augx4spx3_fbank81_sequential/resnet34_se_am
extract_cfg=near_epoch_4
subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh \
	--nj 8 --sliding true --window 1.5 --period 0.75 --min_segment 0.5 --force true --use-gpu true \
	--model 4.params --nnet-config config/near.extract.config \
	$nnet_dir data/$prefix/dihard_2020_dev $nnet_dir/$extract_cfg/dihard_2020_dev
\cp $nnet_dir/$extract_cfg/dihard_2020_dev/subsegments_data/{spk2utt,utt2spk,segments} \
 $nnet_dir/$extract_cfg/dihard_2020_dev
	
subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh \
	--nj 8 --sliding true --window 1.5 --period 0.75 --min-segment 0.5 --force true --use-gpu true \
	--model 4.params --nnet-config config/near.extract.config \
	$nnet_dir data/$prefix/dihard_2020_eval $nnet_dir/$extract_cfg/dihard_2020_eval
\cp $nnet_dir/$extract_cfg/dihard_2020_eval/subsegments_data/{spk2utt,utt2spk,segments} \
 $nnet_dir/$extract_cfg/dihard_2020_eval

#[4]zero-mean normalization for voxceleb2 and dihard2020
dataset=voxceleb2_train
ivector-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp $nnet_dir/$extract_cfg/$dataset/xvector.global.vec
ivector-subtract-global-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark
ivector-normalize-length --scaleup=false ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean_norm.ark

dataset=dihard_2020_dev
ivector-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp $nnet_dir/$extract_cfg/$dataset/xvector.global.vec
ivector-subtract-global-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark
ivector-normalize-length --scaleup=false ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean_norm.ark

dataset=dihard_2020_eval
ivector-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp $nnet_dir/$extract_cfg/$dataset/xvector.global.vec
ivector-subtract-global-mean scp:$nnet_dir/$extract_cfg/$dataset/xvector.scp ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark
ivector-normalize-length --scaleup=false ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean.ark ark:$nnet_dir/$extract_cfg/$dataset/xvector_submean_norm.ark
```

## [3] Convert format

```shell
#[9]prepare voxceleb2(train speaker clustering)、voxceleb1(test speaker clustering) xvector.bin for cdgcn
python local/convert_ark2bin.py --data_dir /work/wj/Extractor_lgcn/data/fbank_81/voxceleb2_train \
 --ivectors_reader /work/wj/Extractor_lgcn/$nnet_dir/$extract_cfg/voxceleb2_train/xvector_submean_norm.ark \
 --save_dir /work/wj/Extractor_lgcn/$nnet_dir/$extract_cfg/voxceleb2_train/lgcn_data \
 --prefix vox2_resnet34se

#dihard dev xvector.bin (dev diarization)
dataset=dihard_2020_dev
python local/convert_ark2bin.py --data_dir $nnet_dir/$extract_cfg/$dataset \
 --ivectors_reader $nnet_dir/$extract_cfg/$dataset/xvector_submean_norm.ark \
 --save_dir $nnet_dir/$extract_cfg/$dataset/lgcn_data \
 --prefix $dataset --diarization
#segments files
python diarization_subtools/split_rttm2single.py $nnet_dir/$extract_cfg/$dataset/segments $nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments
rename '.rttm' '_segments' $nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments/*
dirname=$nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments/
for i in `ls $dirname`;do
	mv -f $dirname/$i $dirname/"dihard_2020_dev_"$i
done 

#dihard eval xvector.bin (eval diarization)
dataset=dihard_2020_eval
python local/convert_ark2bin.py --data_dir $nnet_dir/$extract_cfg/$dataset \
 --ivectors_reader $nnet_dir/$extract_cfg/$dataset/xvector_submean_norm.ark \
 --save_dir $nnet_dir/$extract_cfg/$dataset/lgcn_data \
 --prefix $dataset --diarization
#segments files
python diarization_subtools/split_rttm2single.py $nnet_dir/$extract_cfg/$dataset/segments $nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments
rename '.rttm' '_segments' $nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments/*
dirname=$nnet_dir/$extract_cfg/$dataset/lgcn_data/subsegments/
for i in `ls $dirname`;do
	mv -f $dirname/$i $dirname/"dihard_2020_eval_"$i
done 
```

