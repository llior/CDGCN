#!/usr/bin/env python3
#
#               
#
# Apache 2.0
#
# 
#  -- Modified from Zili Huang's (Johns Hopkins University) VB_resegmentation.py 
#  -- Uses the speaker attribution (q) matrix from VB resegmentation and an overlap 
#  -- hypothesis rttm to assign speakers to overlapped frames 
#
# JSALT 2019, Latané Bullock
# version 2.0 2022.9.29  -- modified by wangjie 2020 xmu
# 2022.10.11 fix-bug if osd rttm is empty
import numpy as np
import argparse
import glob
from tqdm.contrib import tqdm
import matplotlib.pyplot as plt

import os
def get_utt_list(utt2spk_filename):
    with open(utt2spk_filename, 'r') as fh:
        content = fh.readlines()
    utt_list = [line.split()[0] for line in content]
    print("{} utterances in total".format(len(utt_list)))
    return utt_list

# prepare utt2num_frames dictionary
def get_utt2num_frames(utt2num_frames_filename):
    utt2num_frames = {}
    with open(utt2num_frames_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames

def rttm2one_hot(uttname, utt2num_frames, full_rttm_filename, spkId_dic=None):
    num_frames = utt2num_frames[uttname]
    ref = np.zeros(num_frames)
    #spkId_dic = {'A':0}
    if spkId_dic==None:
        spkId_dic = {}
        spkId_val = 1
    else:
        spkId_val = max(spkId_dic.values())+1
    with open(full_rttm_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname_line = line_split[1]
        # if uttname != uttname_line:
        #     continue
        start_time, duration, spkId = int(float(line_split[3]) * 1000), int(float(line_split[4]) * 1000), line_split[7]
        end_time = start_time + duration
        
        for i in range(start_time, end_time):
            if i < 0:
                raise ValueError("Time index less than 0")
            elif i >= num_frames:
                print('rttm extends beyond the number of frames...')
                print(line)
                print('Start time: ', start_time)
                print('End time: ', end_time)
                print('i: ', i) 
                print('num frame: ', num_frames)               
                # raise ValueError("Time index exceeds number of frames")
                break
            else:
                if spkId in spkId_dic.keys():
                    ref[i] = spkId_dic[spkId]
                else:
                    ref[i] = spkId_val
                    spkId_dic.update({spkId: spkId_val})
                    spkId_val += 1

    return ref.astype(int),spkId_dic

# create output rttm file
def create_rttm_output(uttname, pri_sec, predicted_label, output_dir, channel):
    num_frames = len(predicted_label)

    start_idx = 0
    seg_list = []

    last_label = predicted_label[0]
    for i in range(num_frames):
        if predicted_label[i] == last_label: # The speaker label remains the same.
            continue
        else: # The speaker label is different.
            if last_label != 0: # Ignore the silence.
                seg_list.append([start_idx, i, last_label])
            start_idx = i
            last_label = predicted_label[i]
    if last_label != 0:
        seg_list.append([start_idx, num_frames, last_label])

    rttm_dir = "{}/{}".format(output_dir, pri_sec)
    if not os.path.exists(rttm_dir):
        os.makedirs(rttm_dir)

    with open("{}/{}.rttm".format(rttm_dir, uttname), 'w') as fh:
        for i in range(len(seg_list)):
            start_frame = (seg_list[i])[0]
            end_frame = (seg_list[i])[1]
            label = (seg_list[i])[2]
            duration = end_frame - start_frame
            fh.write("SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n".format(uttname, channel, start_frame / 1000.0, duration / 1000.0, label))
    return 0

#according to rttms files，calculate the number ms level frames
def rttm2msFrames(rttms_dir, utt2num_frames_path):

    utt2num_frames_lines = []
    for rttm_path in sorted(glob.glob(rttms_dir+"/*")):
        utt_name = os.path.split(rttm_path)[1].split(".")[0]
        max_time = 0
        with open(rttm_path, "r") as rttm_file:
            for rttmLine in rttm_file.readlines():
                startTime, durationTime = rttmLine.split()[3], rttmLine.split()[4]
                endTime = float(startTime) + float(durationTime)
                if endTime > max_time:
                    max_time = endTime
        max_frames = max_time * 1000
        utt2num_frames_line = " ".join([utt_name, str(round(max_frames)), "\n"])
        utt2num_frames_lines.append(utt2num_frames_line)
    with open(utt2num_frames_path, "w") as utt2num_frames_file:
        utt2num_frames_file.writelines(utt2num_frames_lines)


def main():
    parser = argparse.ArgumentParser(description='Frame-level overlap reassignment with speaker posterior attributions')
    parser.add_argument('osd_dir', type=str, help='Path to directory where we have necessary files for overlap reassignment\n'+
                        'including utt2spk utt2num_frames rttms rttms2nd osd')

    args = parser.parse_args()
    print(args)
    utt2num_frames_path = "{}/utt2num_frames".format(args.osd_dir)
    utt2spk_path = "{}/utt2spk".format(args.osd_dir)
    rttms_dir = '{}/rttms'.format(args.osd_dir)
    utt_list = get_utt_list(utt2spk_path)
    if not os.path.exists(utt2num_frames_path):
        rttm2msFrames(rttms_dir, utt2num_frames_path)
    utt2num_frames = get_utt2num_frames(utt2num_frames_path)
    
    for utt in tqdm(utt_list):
        n_frames = utt2num_frames[utt]
        #spkId_dic contains mapping between speaker and id

        first_label, spkIdMap_dic = rttm2one_hot(utt, utt2num_frames, '{}/rttms/{}.rttm'.format(args.osd_dir, utt))
        second_label, _ = rttm2one_hot(utt, utt2num_frames, '{}/rttms2nd/{}.rttm'.format(args.osd_dir, utt), spkIdMap_dic)
        # unique, counts = np.unique(vad, return_counts=True)
        # voiced_frames = dict(zip(unique, counts))[1]
        if not os.path.exists('{}/osd/{}.rttm'.format(args.osd_dir, utt)):
            print("warning! {} osd file does not exist".format(utt))
            overlap = np.zeros(first_label.shape)
        else:
            overlap, _ = rttm2one_hot(utt, utt2num_frames, '{}/osd/{}.rttm'.format(args.osd_dir, utt))

        # Keep only the voiced frames (0 denotes the silence 
        # frames, 1 denotes the overlapping speech frames).
        mask = (overlap >= 1)

        create_rttm_output(utt, 'masked_rttms2nd', mask*second_label, args.osd_dir, channel=1)
        create_rttm_output(utt, 'rttms1st', first_label, args.osd_dir, channel=1)
        #create 2 spkeaers rttm files
        rttmTowSpk_dir = "{}/rttmsTwoSpk".format(args.osd_dir)
        first_rttm = "{}/rttms1st/{}.rttm".format(args.osd_dir, utt)
        masked_second_rttm = "{}/masked_rttms2nd/{}.rttm".format(args.osd_dir, utt)
        if not os.path.exists(rttmTowSpk_dir):
            os.mkdir(rttmTowSpk_dir)
        os.system("cat {} {} > {}/{}.rttm".format(first_rttm, masked_second_rttm, rttmTowSpk_dir, utt))

    return 0



if __name__ == "__main__":
    main()
