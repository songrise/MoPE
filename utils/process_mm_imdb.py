#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os


def format_mmimdb_dataset(dataset_root_path):
    train_label_set = set()
    is_save_sample = True
    with open(os.path.join(dataset_root_path, "split.json")) as fin:
        data_splits = json.load(fin)
    for split_name in data_splits:
        with open(os.path.join(dataset_root_path, split_name + ".jsonl"), "w") as fw:
            for idx in data_splits[split_name]:
                with open(os.path.join(dataset_root_path, "dataset/{}.json".format(idx))) as fin:
                    data = json.load(fin)
                plot_id = np.array([len(p) for p in data["plot"]]).argmax()
                dobj = {}
                dobj["id"] = idx
                dobj["text"] = data["plot"][plot_id]
                dobj["img"] = "dataset/{}.jpeg".format(idx)
                dobj["label"] = data["genres"]
                # dele the genres that are not described in the paper
                if "News" in dobj["label"]:
                    print("Skip News")
                    continue
                if "Reality-TV" in dobj["label"]:
                    print("Skip Reality-TV")
                    continue
                if "Talk-Show" in dobj["label"]:
                    print("Skip Talk-Show")
                    continue
                if split_name == "train":
                    for label in dobj["label"]:
                        train_label_set.add(label)
                else:
                    for label in dobj["label"]:
                        if label not in train_label_set:
                            is_save_sample = False
                if len(dobj["text"]) > 0 and is_save_sample:
                    fw.write("%s\n" % json.dumps(dobj))
                is_save_sample = True


def construct_labels_file(dataset_root_path):
    """
    get all class lables
    """
    labels = []
    titles_file = os.path.join(dataset_root_path, "texts","test_titles.csv")
    with open(titles_file, "r") as f:
        # for each line get the 3rd comma separated value
        for line in f:
            labels.append(line.split(",")[2])
    labels = list(set(labels))
    labels.sort(key=lambda x: x.lower())
    with open(os.path.join(dataset_root_path, "labels.txt"), "w") as f:
        for label in labels:
            f.write(label)
    print("done")

if __name__ == "__main__":
    # Path to the directory for MMIMDB
    path = "/root/autodl-tmp/PromptDistill/data/mmimdb/"
    format_mmimdb_dataset(path)
    # construct_labels_file(path)
