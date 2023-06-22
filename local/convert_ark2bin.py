import scipy
import numpy as np
# import cupy as cp
import argparse
import math
import os
import random
import operator
import collections

import sys

# from plda import PldaStats,PldaEstimation
sys.path.insert(0, ".")
import utils.kaldi_io as kaldi_io


# /tsdata/kaldi/egs/tfc/gan/exp/Xmu_voxceleb1_spec_am_reduceP_sgd_tdnn_sequential_multi_resolution/far_epoch_15/voxceleb1_train_aug_2fold/xvector_submean_norm_lda256_submean_norm.ark
# Copyright xmuspeech （Author:wangjie 2022-8-7)

def get_args():
    parser = argparse.ArgumentParser(
            description="""Transform Kaldi's file to expected data(lgcn format data).""")

    parser.add_argument("--data_dir", type=str,help="kaldi data directory including The spk2utt, utt2spk, subesgments file.")
    parser.add_argument("--ivectors_reader", type=str, help="kaldi format features file (xvector.ark)")
    parser.add_argument("--save_dir", type=str, help="lgcn format features file and label. The directory contain vox1.bin")
    parser.add_argument("--prefix", type=str, help="prefix of dataset. For example vox1_tdnn")
    parser.add_argument("--diarization", action="store_true", help="if diarization is true, we extract the slide windows embedding for each utterance")

    args = parser.parse_args()
    return args


def main():
    if not diarizaton:
        spkid_dict = {}
        label_lines = []
        label = -1
        with open(utt2spk, 'r') as f:
            for utt2spk_line in f.readlines():
                spk_id = utt2spk_line.split()[1]
                if spk_id not in spkid_dict:
                    spkid_dict[spk_id] = label
                    label += 1
                label_lines.append(str(label)+"\n")
            size = len(label_lines)

        with open(label_path, "w") as label_file:
            label_file.writelines(label_lines)

        utt2spk_dict = {}

        with open(spk2utt, 'r') as f:
            for line in f:
                temp_list = line.strip().split()
                spk = temp_list[0]

                del temp_list[0]
                for utt in temp_list:
                    utt2spk_dict[utt] = spk

        spk2vectors = {}

        for key, vector in kaldi_io.read_vec(ivectors_reader):
            try:
                spk = utt2spk_dict[key]
            except KeyError:
                print("Warning,segment id {} not in utt2spk file. You may filter segments".format(key))
                continue
            try:
                tmp_list = spk2vectors[spk]
                tmp_list.append(vector)
                spk2vectors[spk] = tmp_list
            except KeyError:
                spk2vectors[spk] = [vector]

        spk2vectors_new = collections.OrderedDict(sorted(spk2vectors.items()))

        data = list()
        print("Data loading... ")
        for key in spk2vectors_new.keys():
            vectors = np.array(spk2vectors_new[key], dtype=np.float32)
            data.append(vectors)

        pooled_data = np.vstack(data)

        print(pooled_data.shape[0])
        print(size)
        assert pooled_data.shape[0] == size
        pooled_data.tofile(features_path)
    else:
        #seg_dict={'DH_DEV_0001_0000-00000000-00000054':('DH_DEV_0001', seg_id_offset)}
        #where 'DH_DEV_0001_0000-00000000-00000054' is segment id that belong to DH_DEV_0001 meeting, 'seg_id_offset' is the index of vector
        seg_dict = {}
        #visited_meetings={'DH_DEV_0001':segment_num}
        #where segment_num is the total number of the meeting
        visited_meetings = dict()
        seg_id_offset = -1
        with open(utt2spk, "r") as utt2spk_file:
            for utt2spk_line in utt2spk_file.readlines():
                seg_id, meeting_id = utt2spk_line.split()
                if meeting_id in visited_meetings:
                    seg_id_offset += 1
                else:
                    seg_id_offset = 0
                visited_meetings[meeting_id] = seg_id_offset+1
                seg_dict[seg_id] = (meeting_id, seg_id_offset)

        meetings_feature = {}
        features_dim = None
        for seg_id, vector in kaldi_io.read_vec(ivectors_reader):
            features_dim = len(vector)
            try:
                meeting_id = seg_dict[seg_id][0]
            except KeyError:
                print("Warning,segment id {} not in utt2spk file. You may filter segments".format(seg_id))
                continue
            if meeting_id not in meetings_feature:
                meetings_feature[meeting_id] = np.zeros((visited_meetings[meeting_id], features_dim), dtype=np.float32)

            cur_vector_offset = seg_dict[seg_id][1]
            meetings_feature[meeting_id][cur_vector_offset] = vector

        #create features.bin and labels.meta files
        for meeting_id, vectors in meetings_feature.items():
            cur_features_path = os.path.join(save_dir, "features/{}_{}.bin".format(prefix, meeting_id))
            cur_labels_path = os.path.join(save_dir, "labels/{}_{}.meta".format(prefix, meeting_id))
            pooled_data = vectors.reshape(-1,features_dim)
            pooled_data.tofile(cur_features_path)
            with open(cur_labels_path, "w") as cur_labels_file:
                cur_labels_file.write("0\n"*len(vectors))

        with open(meetingList_path,"w") as meetingList_file:
            for key in meetings_feature.keys():
                meetingList_file.write("_".join([prefix,key])+"\n")
'''
for example:
    spk2utt = "/work/wj/Extractor_lgcn/data/mfcc_40/voxceleb1_train/spk2utt"
    utt2spk = "/work/wj/Extractor_lgcn/data/mfcc_40/voxceleb1_train/utt2spk"
    ivectors_reader = "/work/wj/Extractor_lgcn/exp/vox1_augx4_mfcc40_noise/extended_baseline/near_epoch_15/voxceleb1_train/xvector_submean_norm.ark"
    save_dir = "/work/wj/Extractor_lgcn/exp/vox1_augx4_mfcc40_noise/extended_baseline/near_epoch_15/voxceleb1_train/lgcn_data/vox1_tdnn.bin"
    label_path = "/work/wj/Extractor_lgcn/exp/vox1_augx4_mfcc40_noise/extended_baseline/near_epoch_15/voxceleb1_train/lgcn_data/vox1_tdnn.meta"
'''
if __name__ == "__main__":
    args = get_args()

    spk2utt = os.path.join(args.data_dir, "spk2utt")
    utt2spk = os.path.join(args.data_dir, "utt2spk")
    segments = os.path.join(args.data_dir, "segments")
    ivectors_reader = args.ivectors_reader
    save_dir = args.save_dir
    prefix = args.prefix
    features_path = os.path.join(save_dir,"features/{}.bin".format(prefix))
    label_path = os.path.join(save_dir,"labels/{}.meta".format(prefix))
    meetingList_path = os.path.join(save_dir,"{}.list".format(prefix))
    diarizaton = args.diarization
    if not os.path.exists(os.path.join(save_dir, "features")):
        os.makedirs(os.path.join(save_dir, "features"))
    if not os.path.exists(os.path.join(save_dir, "labels")):
        os.makedirs(os.path.join(save_dir, "labels"))
    main()



