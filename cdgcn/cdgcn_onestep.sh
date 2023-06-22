#copyright wangjie xmuspeech 2022.08.11
##########################################################################k100
#prepare knn graph
```
prefix=./data_dihard
name=dihard_2020_dev_DH_DEV_0001
cfg_name=cfg_train_lgcn_tdnn_dihard_2020
model_name=cfg_train_lgcn_tdnn_vox1_k100_200k1_20k2_20u
```
stage='prediction'
. ./parse_options.sh


prefix=$1
name=$2
cfg_name=$3
model_name=$4
cd_method=$5
knn=$6
dim=$7
second_out=$8

set -e
if [ ! $# -eq 8 ];then
	echo "$0 the number of parameter(=$#) is error."
	exit 1
fi

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

export PYTHONPATH=.
knn_method="faiss"
echo ">>>>>>>>>>>>>>"
if [ ! -d $prefix/knns/$name ]; then
	mkdir -p $prefix/knns/$name
	OMP_NUM_THREADS=1 python tools/create_knn.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
	--knn_method $knn_method \
    --knn $knn > $prefix/knns/$name/create_knn.log
fi

	
#######################k100 200k1 20k2 20u

config=cdgcn/configs/$cfg_name.py

# train
#if [$stage=="training"]; then
# python lgcn/main.py \
#  --config $config \
#  --phase 'train'
#fi
# test

if [ $stage=="prediction" ]; then
 
 load_from=$prefix/work_dir/$model_name/latest.pth
 
 python cdgcn/main.py \
     --config $config \
     --phase 'test' \
     --load_from $load_from \
     --save_output \
	 --diarization \
	 --test_name $name

 #make the diartory of labels
 if [ ! -d $prefix/work_dir/$model_name ]; then
	mkdir -p $prefix/work_dir/$model_name
 fi
 #create label file for meeting
 awk '{print $1}' $prefix/subsegments/${name}_segments | \
 paste - $prefix/work_dir/$model_name/${cd_method}_labels/${name}_pred_labels.txt > \
 $prefix/work_dir/$model_name/${cd_method}_labels/${name}.label
 
 #convert label to rttm files
 diarization/make_rttm.py --rttm-channel 1 $prefix/subsegments/${name}_segments $prefix/work_dir/$model_name/${cd_method}_labels/${name}.label \
 $prefix/work_dir/$model_name/${cd_method}_labels/${name}.rttm
 
 if [ $second_out=="True" ]; then
	 #create secondary speaker label file for meeting
	 awk '{print $1}' $prefix/subsegments/${name}_segments | \
	 paste - $prefix/work_dir/$model_name/${cd_method}_labels/secondary/${name}.txt > \
	 $prefix/work_dir/$model_name/${cd_method}_labels/secondary/${name}.label
	 
	 #convert label to rttm files
	 second_rttm_path=$prefix/work_dir/$model_name/${cd_method}_labels/secondary/${name}.rttm
	 diarization/make_rttm.py --rttm-channel 1 $prefix/subsegments/${name}_segments $prefix/work_dir/$model_name/${cd_method}_labels/secondary/${name}.label \
	 $second_rttm_path
	 #remove non-overlap label
	 #awk '$8!="-1"{print $0}' $second_rttm_path.tmp > $second_rttm_path
	 #merge first label and secondary label
	 overlapped_dir=$prefix/work_dir/$model_name/${cd_method}_labels/overlapped_rttm
	 if [ ! -d $overlapped_dir ]; then
		mkdir -p $overlapped_dir
	 fi
	 cat $prefix/work_dir/$model_name/${cd_method}_labels/${name}.rttm $second_rttm_path > $overlapped_dir/${name}.rttm
 fi

fi

 