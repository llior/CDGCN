rirs_path=/tsdata/sre/RIRS_NOISES
musan_path=/tsdata/sre/musan



############################################# Start ##################################################

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
# subtools/newCopyData.sh mfcc_40 "data/voxceleb2_train_spx3_augx2 data/voxceleb2_train"
# subtools/newCopyData.sh mfcc_40_pitch "data/mfcc_40/voxceleb1_train data/mfcc_40/voxceleb1_train_speedx5_ori_label voxceleb1_test_speedx5_speed1.0/voxceleb1_test_speedx5_speed1.0"

# # # Augment trainset by clean:aug=1:1 with Kaldi augmentation (randomly select the same utts of clean dataset
# # # from reverb, noise, music and babble copies.)

subtools/augmentDataByNoise.sh --rirs_noises $rirs_path --musan $musan_path --factor 4 \
                                --reverb true --noise true --music true --babble true \
                               data/mfcc_40/voxceleb1_test  data/mfcc_40/voxceleb1_test_augx4
# exit 1;

# subtools/kaldi/utils/perturb_data_dir_speed.sh 0.8 data/mfcc_40/voxceleb1_train data/mfcc_40/voxceleb1_train_sp0.8
# subtools/kaldi/utils/perturb_data_dir_speed.sh 0.9 data/mfcc_40/voxceleb1_train data/mfcc_40/voxceleb1_train_sp0.9
# subtools/kaldi/utils/perturb_data_dir_speed.sh 1.1 data/mfcc_40/voxceleb1_train data/mfcc_40/voxceleb1_train_sp1.1
# subtools/kaldi/utils/perturb_data_dir_speed.sh 1.2 data/mfcc_40/voxceleb1_train data/mfcc_40/voxceleb1_train_sp1.2
# exit()
# subtools/kaldi/utils/data/perturb_data_dir_speed_3way.sh data/voxceleb2_train data/voxceleb2_train_spx3
# subtools/kaldi/utils/data/perturb_data_dir_speed_5way.sh data/mfcc_40/voxceleb1_test data/mfcc_40/voxceleb1_test_speedx5
# exit 1;
# subtools/kaldi/utils/combine_data.sh data/fbank_81/voxceleb2_train_augx4_spx3_nosil data/fbank_81/voxceleb2_train_spx3_nosil data/fbank_81/voxceleb2_train_only_augx4_nosil
# exit 1;

# # Make features for trainset


subtools/makeFeatures.sh --pitch true --pitch-config  subtools/conf/pitch.conf \
                                 --nj 40 --exp exp/features --use_gpu false \
                               /tsdata/kaldi/egs/tfc/gan/data/fbank_80/voxceleb1_test fbank  conf/sre-fbank-81.conf
# # exit 1;
# # prefix=fbank_81
# # subtools/filterDataDir.sh --split-aug false /tsdata/kaldi/egs/tfc/gan/data/fbank_81/voxceleb2_train_augx4_spx3_nosil/ /tsdata/kaldi/egs/tfc/gan/data/fbank_81/voxceleb2_train/utt2spk \
# #                                             /tsdata/kaldi/egs/tfc/gan/data/fbank_81/voxceleb2_train_nosil/
# # exit 1;


# subtools/computeAugmentedVad.sh --nj 40  data/mfcc_80/voxceleb2_train_augx4 \
#                                 data/mfcc_80/voxceleb2_train/utt2spk conf/vad-5.5.conf

# exit 1;
subtools/computeVad.sh --nj 40 /tsdata/kaldi/egs/tfc/gan/data/fbank_80/voxceleb1_test conf/vad-5.5.conf
exit 1;
# Make features for testset

# subtools/makeFeatures.sh --pitch false --pitch-config subtools/conf/pitch.conf \
#                                  --nj 80 \
#                                 --exp exp/features data/mfcc_80/voxceleb1_test \
#                                     mfcc    conf/sre-mfcc-80.conf

# subtools/computeVad.sh --nj 80 data/mfcc_80/voxceleb1_test conf/vad-5.5.conf

# subtools/makeFeatures.sh  --exp exp_back/features data/fbank_80/task2_indomain_train_aug fbank \
#                                 subtools/conf/sre-fbank-80.conf

# subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf --exp exp/features data/mfcc_23_pitch/voxceleb1_train_noise mfcc\
#                                 subtools/conf/sre-mfcc-23.conf
# subtools/computeVad.sh data/mfcc_23_pitch/voxceleb1_train_noise subtools/conf/vad-5.5.conf
# subtools/makeFeatures.sh  --exp exp_back/features data/fbank_80/task2_indomain_train_aug fbank \
#                                 subtools/conf/sre-fbank-80.conf

# Compute VAD for augmented trainset
# subtools/computeAugmentedVad.sh data/fbank_80/task2_indomain_train_aug data/fbank_80/task2_indomain_train/utt2spk \
#                                 subtools/conf/vad-5.5.conf
error=5
/bin/cp -r /tsdata/kaldi/egs/tfc/gan/exp/egs/mfcc40_voxceleb2_train_sequential/valid.egs.csv /tsdata/kaldi/egs/tfc/gan/exp/egs/mfcc40_voxceleb2_train_error${error}_sequential/
# /bin/cp -r /tsdata/kaldi/egs/tfc/gan/data/fbank_81/voxceleb2_train_error${error}_nosil/utt2spk /tsdata/kaldi/egs/tfc/gan/data/mfcc_40/voxceleb2_train_error${error}_nosil/



exit 1;
# Make features for testset
# subtools/makeFeatures.sh  --exp exp_back/features data/fbank_80/task2_enroll fbank \
#                                 subtools/conf/sre-fbank-80.conf

# # Compute VAD for testset which is clean
# subtools/computeVad.sh data/fbank_80/task2_enroll subtools/conf/vad-5.5.conf




# subtools/makeFeatures.sh  --exp exp_back/features data/fbank_80/task2_evaluation fbank \
#                                 subtools/conf/sre-fbank-80.conf

# subtools/computeVad.sh data/fbank_80/task2_evaluation/ subtools/conf/vad-5.5.conf





####################### Training (preprocess -> get_egs -> training -> extract_xvectors)##################

# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
# Both two launchers just train a standard x-vector baseline system and other methods like multi-gpu training, extended xvector, 
# AM-softmax loss etc. could be set by yourself. 
# subtools/runPytorchLauncher.sh subtools/recipe/voxceleb/runSnowdarXvector-voxceleb1.py --stage=0
# subtools/runPytorchLauncher.sh subtools/recipe/voxceleb/runSnowdarXvector-voxceleb1o2.py --stage=0