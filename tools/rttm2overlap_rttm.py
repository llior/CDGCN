#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Authors: wanjie xmuspeech

import os
import argparse
from pyannote.database.util import load_rttm

def write_segments(segTimeline, segment_file):
    utt_id = segTimeline.uri
    index = 0
    for seg in segTimeline.for_json()["content"]:
        seg_id = utt_id + "_" + str(index).zfill(4)
        segLine = "{} {} {:.3f} {:.3f}\n".format(seg_id, utt_id, seg['start'], seg['end'])
        segment_file.write(segLine)
        index += 1

def rttm2oneHot(rttmTimeline):
    print("print")

if __name__ == "__main__":

    rttm_path = "/work/wj/learn-to-cluster/data_dihard/ref.rttm"
    ovl_rttm_path = "/work/wj/learn-to-cluster/data_dihard/ovl_rttm"

    ovl_rttmTmp_dir = os.path.dirname(ovl_rttm_path)+"/tmp"

    rttmTimeline = load_rttm(rttm_path)
    if not os.path.exists(ovl_rttmTmp_dir):
        os.makedirs(ovl_rttmTmp_dir)

    for utt_id in rttmTimeline.keys():
        ovlTimeline = rttmTimeline[utt_id].get_timeline().get_overlap()


        with open(os.path.join(ovl_rttmTmp_dir, utt_id), "w") as ovl_rttm_file:
            ovlTimeline.to_annotation().write_rttm(ovl_rttm_file)

    os.system("cat {}/* > {}".format(ovl_rttmTmp_dir, ovl_rttm_path))





    pass
