#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author    wangjie xmuspeech
#split total rttm file into singel rttm file
import os
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('rttm_path',type=str, help="input rttm file path")
    parser.add_argument('rttms_dir', type=str, help="output splited rttm files directory")


    args = parser.parse_args()
    rttm_path = args.rttm_path
    rttms_dir = args.rttms_dir



    utt_dict = {}
    with open(rttm_path,"r") as rttm_file:
        for rttm_line in rttm_file.readlines():
            utt_id = rttm_line.split()[1]
            if utt_id not in utt_dict.keys():
                utt_dict.update({utt_id: [rttm_line]})
            else:
                utt_dict[utt_id].append(rttm_line)

    if not os.path.exists(rttms_dir):
        os.makedirs(rttms_dir)

    for utt_id in utt_dict.keys():
        single_rttm_path = os.path.join(rttms_dir, utt_id+".rttm")
        with open(single_rttm_path, "w") as single_rttm_file:
            single_rttm_file.writelines(utt_dict[utt_id])

    pass

