# Run CDGCN

### [1]Train CDGCN model

stage=''
. ./parse_options.sh

export PYTHONPATH=.
prefix=./xv_data/resnet34se
name=./vox2_resnet34se
cfg_name=cfg_cdgcn_vox2
model_name=cfg_cdgcn_vox2
config=cdgcn/configs/$cfg_name.py
cd_method="Leiden"
dim=256
gpu_num=1
knn=300

#multi-gpu 
if [ $stage == "Train" ]; then

	OMP_NUM_THREADS=1 python tools/create_knn.py \
    --prefix $prefix \
    --name $name \
    --dim $dim \
    --knn_method "faiss_gpu" \
    --knn $knn 
	
	if [ $gpu_num -gt 1 ]; then
		NCCL_P2P_LEVEL=NVL python launch.py --use_env --nproc_per_node $gpu_num cdgcn/main.py   --phase train     --config $config --gpus $gpu_num --distributed
	else
		#single-gpu
		python cdgcn/main.py   --phase train     --config $config
	fi
	
fi

### [2]CDGCN 1st speaker label
if [ $stage == "Test_1st" ]; then

	for name in `cat xv_data/dihard_2020_dev.list`;do
		./cdgcn/cdgcn_onestep.sh $prefix $name $cfg_name $model_name $cd_method 300 256 True
	done

	cat $prefix/work_dir/$model_name/${cd_method}_labels/*.rttm > $prefix/work_dir/$model_name/sys.rttm
	sys=$prefix/work_dir/$model_name/sys.rttm
	diarization/md-eval.pl -s $sys -r xv_data/ref_dev.rttm
fi


### [3]CDGCN Graph-OSD
if [ $stage == "Test_2nd" ]; then

	FILE_LIST=xv_data/dihard_2020_dev.list
	exp_dir=xv_data/resnet34se/work_dir/cfg_cdgcn_vox2/Leiden_labels
	osd_rttm_dir=xv_data/osd/model/pyannote2.0/dev/rttm
	osd_dir=$exp_dir/overlap_aware
	rm -rf $osd_dir/osd
	if [ ! -d $osd_dir/osd ]; then
		mkdir -p $osd_dir/osd
	fi
	awk '{print $1,$1}' $FILE_LIST > $osd_dir/utt2spk

	\cp $osd_rttm_dir/* $osd_dir/osd
	rename 'DH' 'dihard_2020_dev_DH' $osd_dir/osd/*
	#1st speaker label
	if [ ! -d $osd_dir/rttms ]; then
		mkdir -p $osd_dir/rttms
	fi
	\cp $exp_dir/*.rttm $osd_dir/rttms
	#2nd speaker label
	if [ ! -d $osd_dir/rttms2nd ]; then
		mkdir -p $osd_dir/rttms2nd
	fi
	\cp $exp_dir/secondary/*.rttm $osd_dir/rttms2nd/

	python cdgcn/diar_ovl_assignment_ms.py  $osd_dir
	cat $osd_dir/rttmsTwoSpk/* > $osd_dir/sysTwoSpk.rttm
	sed -i 's/dihard_2020_dev_//' $osd_dir/sysTwoSpk.rttm 
	#calculate DER
	if [ ! -d $osd_dir/result ];then
		 mkdir -p $osd_dir/result
	fi
	sys=$osd_dir/sysTwoSpk.rttm

	diarization/md-eval.pl -r xv_data/ref_dev.rttm -s $sys 
	
fi