import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import json
import torchvision as tv
import  json

import pyarrow as pa
from PIL import Image
import librosa
import io


# Torch libraries
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

IMG_MEAN = [0.46777044, 0.44531429, 0.40661017]
IMG_STD = [0.12221994, 0.12145835, 0.14380469]


def line_to_paths_fn_nyudv2(x, input_names):
    return x.decode("utf-8").strip("\n").split("\t")


line_to_paths_fn = {"nyudv2": line_to_paths_fn_nyudv2}


class LowShotDataset(Dataset):
    def __init__(self, dataset, n_shot):
        self.dataset = dataset
        self.n_shot = n_shot
        self.original_length = len(dataset)
        # TODO Nov 04: fix the random index
        self.select_idx = np.random.choice(
            self.original_length, self.n_shot, replace=False
        )

    def __getitem__(self, index):
        return self.dataset[self.select_idx[index]]

    def __len__(self):
        return self.n_shot


class SegDataset(Dataset):
    """Multi-Modality Segmentation dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        line_to_paths_fn (callable): function to convert a line of data_file
            into paths (img_relpath, msk_relpath, ...).
        masks_names (list of strings): keys for each annotation mask
                                        (e.g., 'segm', 'depth').
        transform_trn (callable, optional): Optional transform
            to be applied on a sample during the training stage.
        transform_val (callable, optional): Optional transform
            to be applied on a sample during the validation stage.
        stage (str): initial stage of dataset - either 'train' or 'val'.

    """

    def __init__(
        self,
        dataset,
        data_file,
        data_dir,
        input_names,
        input_mask_idxs,
        transform_trn=None,
        transform_val=None,
        stage="train",
        ignore_label=None,
    ):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn[dataset](l, input_names) for l in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.input_names = input_names
        self.input_mask_idxs = input_mask_idxs
        self.ignore_label = ignore_label

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        idxs = self.input_mask_idxs
        names = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {}
        for i, key in enumerate(self.input_names):
            sample[key] = self.read_image(names[idxs[i]], key)
        try:
            mask = np.array(Image.open(names[idxs[-1]]))
        except FileNotFoundError:  # for sunrgbd
            path = names[idxs[-1]]
            num_idx = int(path[-10:-4]) + 5050
            path = path[:-10] + "%06d" % num_idx + path[-4:]
            mask = np.array(Image.open(path))
        assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
        sample["inputs"] = self.input_names
        sample["mask"] = mask
        if self.stage == "train":
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == "val":
            if self.transform_val:
                sample = self.transform_val(sample)
        del sample["inputs"]
        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == "depth":
            img = cv2.applyColorMap(
                cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET
            )
        return img

    @staticmethod
    def read_image(x, key):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        img_arr = np.array(Image.open(x))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr


# class UPMCFood101(Dataset):
#     """
#     UPMC Food-101 dataset
#     """

#     def __init__(self, data_root, split='train', transform=None):
#         self.data_root = data_root
#         self.split = split
#         self.transform = transform
#         self.imgs = []
#         self.labels = []

import librosa


import json
import torch
import librosa
from torch.utils.data import Dataset


class MMSDDataset(Dataset):
    def __init__(self, data_root, split):
        annotation_path = f"{data_root}/sarcasm_data.json"
        self.data_root = data_root
        self.annotation = json.load(open(annotation_path))
        # Convert to list
        self.annotation_key = [k for k in self.annotation]
        self.annotation_value = [self.annotation[k] for k in self.annotation]

        # train_idx = np.zeros(len(self.annotation_key), dtype=np.int32)
        # # randomly fill 334 True values as train_idx
        # selected_idx = np.random.choice(len(self.annotation_key), 334, replace=False)
        # train_idx[selected_idx] = 1
        # if split == "train":
        #     self.annotation_key = self.annotation_key[train_idx == 1]
        #     self.annotation_value = self.annotation_value[train_idx == 1]
        # elif split == "val":
        #     self.annotation_key = self.annotation_key[train_idx == 0]
        #     self.annotation_value = self.annotation_value[train_idx == 0]

        if split == "train":
            self.annotation_key = self.annotation_key[:334]
            self.annotation_value = self.annotation_value[:334]
        elif split == "test" or split == "val":
            self.annotation_key = self.annotation_key[334:]
            self.annotation_value = self.annotation_value[334:]

        # Preload and store all text and wav in memory
        self.data = []
        for key, value in zip(self.annotation_key, self.annotation_value):
            wav_path = f"{self.data_root}/utterances_final/{key}.wav"
            context_text, utterance_text, sarcasm = (
                value["context"],
                value["utterance"],
                value["sarcasm"],
            )
            context_speaker, utterance_speaker = (
                value["context_speakers"],
                value["speaker"],
            )

            # Concatenate speakers to the text
            for i in range(len(context_text)):
                context_text[i] = context_speaker[i] + ": " + context_text[i]
            utterance_text = utterance_speaker + ": " + utterance_text
            # Load wav
            wav, sr = librosa.load(wav_path, sr=16000)
            # cast to half precision
            wav = wav.astype(np.float16)

            sarcasm = torch.LongTensor([sarcasm])

            self.data.append(
                {
                    "id": key,
                    "context": context_text,
                    "utterance": utterance_text,
                    "sarcasm": sarcasm,
                    "wav": wav,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # context_concat = " ".join(item["context"])
        # text = context_concat + " " + item["utterance"]
        text = item["utterance"]
        wav = item["wav"]
        return text, wav, item["sarcasm"]


class JsonlDataset(Dataset):
    def __init__(self, data_root, split: str = "train", transforms=None):
        data_path = os.path.join(data_root, f"{split}.jsonl")
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        if "food" in data_root:
            self.data_type = "uni_label"
        elif "imdb" in data_root:
            self.data_type = "multi_label"
        elif "snli" in data_root:
            self.data_type = "visual_entailment"

        # self.n_classes = n_classes
        self.text_start_token = ["[CLS]"]
        with open(os.path.join(self.data_dir, "labels.txt")) as f:
            self.labels = [l.strip() for l in f.readlines()]
        self.n_classes = len(self.labels)
        # with numpy_seed(0):
        #     for row in self.data:
        #         if np.random.random() < args.drop_img_percent:
        #             row["img"] = None

        self.max_seq_len = 512  #!HARDCODED 

        self.transforms = tv.transforms.ToTensor()
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO Sep 30: check here
        # https://github.com/facebookresearch/mmbt/blob/master/mmbt/data/dataset.py
        if self.data_type != "visual_entailment":
            sentence = self.data[index]["text"]
        else:
            # TODO Sep 30: check snli use sentence 1 or not
            sentence = self.data[index][
                "sentence2"
            ]  # sentence 2 is the text hypothesis

        if self.data_type == "uni_label":
            label = torch.LongTensor([self.labels.index(self.data[index]["label"])])
        elif self.data_type == "multi_label":
            label = torch.zeros(self.n_classes)
            label[[self.labels.index(tgt) for tgt in self.data[index]["label"]]] = 1
        elif self.data_type == "visual_entailment":
            label = torch.LongTensor(
                [self.labels.index(self.data[index]["gold_label"])]
            )

        if self.data_type != "visual_entailment":
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"])
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        else:
            flicker_id = self.data[index]["Flickr30K_ID"]
            img_path = os.path.join(
                self.data_dir, "Flickr30K", "flickr30k_images", flicker_id + ".jpg"
            )
            image = Image.open(img_path).convert("RGB")

        image = self.transforms(image)

        return sentence, image, label


def create_loaders(data_root, batch_size, num_workers, n_shot=0):
    """
    Args:

    Returns:
      train_loader, val loader, test loader

    """
    # no aug

    # Training transformations
    if "food" in data_root:
        transforms_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

    elif "imdb" in data_root:
        transforms_train = transforms.Compose(
            [
                transforms.RandAugment(2, 7),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_val = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_test = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )
    elif "snli" in data_root:
        transforms_train = transforms.Compose(
            [
                transforms.RandAugment(2, 5),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_val = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

        transforms_test = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,
                    std=IMG_STD,
                ),
            ]
        )

    # Training and validation sets
    trainset = JsonlDataset(data_root, "train", transforms_train)
    validset = JsonlDataset(data_root, "val", transforms_val)
    testset = JsonlDataset(data_root, "test", transforms_test)
    if n_shot > 0:
        trainset = LowShotDataset(trainset, n_shot)
    print(
        "Created train set from {}, {} examples, val set {}, test set {} examples".format(
            data_root, len(trainset), len(validset), len(testset)
        )
    )
    # Training and validation loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader


def create_mmsd_loaders(data_root, batch_size, num_workers, n_shot=0):
    """
    Args:

    Returns:
    train_loader, val loader, test loader

    """

    # no aug
    def collate_fn(batch):
        # Extract wav and find max length
        wavs = [item[1] for item in batch]
        max_length = max(len(wav) for wav in wavs)

        # Pad wavs and create a tensor
        padded_wavs = [
            np.pad(wav, (0, max_length - len(wav)), "constant") for wav in wavs
        ]
        padded_wavs = torch.tensor(padded_wavs)

        # Collect other fields
        texts = [item[0] for item in batch]
        sarcasms = torch.stack([item[3] for item in batch])
        frames = [item[2] for item in batch]
        frames = torch.stack(frames).permute(0, 2, 1, 3, 4)
        return texts, padded_wavs, frames, sarcasms

    # frame_path = f"{data_root}/utterances_final/frames/processed_frames.pt"
    # # Load frames
    # frames = torch.load(frame_path, map_location="cpu")
    trainset = MMSDDataset(data_root, "train")
    testset = MMSDDataset(data_root, "test")
    # Training and validation loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

info = {
    "refcoco": {
        "train": 42404,
        "val": 3811,
        "val-test": 3811,
        "testA": 1975,
        "testB": 1810,
    },
    "refcoco+": {
        "train": 42278,
        "val": 3805,
        "val-test": 3805,
        "testA": 1975,
        "testB": 1798,
    },
    "refcocog_u": {"train": 42226, "val": 2573, "val-test": 2573, "test": 5023},
    "refcocog_g": {"train": 44822, "val": 5000, "val-test": 5000},
}


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(
        self,
        lmdb_dir,
        mask_dir,
        dataset,
        split,
        mode,
        transforms,
        input_size,
    ):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.mean = torch.tensor(IMG_MEAN).reshape(3, 1, 1)
        self.std = torch.tensor(IMG_STD).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.transforms = transforms
        self.env = None
        # self._init_db()

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b"__len__"))
            self.keys = loads_pyarrow(txn.get(b"__keys__"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref["img"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref["seg_id"]
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + ".png")
        # sentences
        idx = np.random.choice(ref["num_sents"])
        sents = ref["sents"]
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.46777044 * 255, 0.44531429 * 255, 0.40661017 * 255],
        )
        if self.mode == "train":
            # mask transform
            mask = cv2.imdecode(
                np.frombuffer(ref["mask"], np.uint8), cv2.IMREAD_GRAYSCALE
            )
            mask = cv2.warpAffine(
                mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.0
            )
            mask = mask / 255.0
            # sentence -> vector
            sent = sents[idx]
            img, mask = self.convert(img, mask)
            if self.transforms is not None:
                img = self.transforms(img)
            return img, sent, mask
        elif self.mode == "val":
            # sentence -> vector
            sent = sents[0]
            img = self.convert(img)[0]
            params = {
                "mask_dir": mask_dir,
                "inverse": mat_inv,
                "ori_size": np.array(img_size),
            }
            return img, sent, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                "ori_img": ori_img,
                "seg_id": seg_id,
                "mask_dir": mask_dir,
                "inverse": mat_inv,
                "ori_size": np.array(img_size),
                "sents": sents,
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array(
            [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]],
            np.float32,
        )

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.0).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + f"db_path={self.lmdb_dir}, "
            + f"dataset={self.dataset}, "
            + f"split={self.split}, "
            + f"mode={self.mode}, "
            + f"input_size={self.input_size}, "
        )


class BaseVQADataset(Dataset):
    """
    impl from https://github.com/liangsheng02/Modular-and-Parameter-Efficient-Multimodal-Fusionwith-Prompting/blob/main/mydatasets/base_dataset.py#L11
    """

    def __init__(
        self,
        data_dir: str,
        # transform_keys: list,
        # image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        transform=None,
        subsample_scale=1.0,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        # transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        # assert len(transform_keys) >= 1
        super().__init__()

        # self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.transform = transform

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        if self.transform is not None:
            image = self.transform(image)
        # image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image,
            # "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        return {
            "text": text,
            # "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        return {f"false_text_{rep}": text}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret


class VQAv2Dataset(BaseVQADataset):
    def __init__(self, *args, split="", answer_filter: str = None, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train"]
        elif split == "val":
            names = ["vqav2_val"]
        elif split == "test":
            names = ["vqav2_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        if self.transform is not None:
            # append normalization
            self.transform = tv.transforms.Compose(
                [
                    *self.transform.transforms,
                    tv.transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
                ]
            )
        else:
            self.transform = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.Resize((224, 224)),
                    tv.transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
                ]
            )

        # id2datum
        crt_global_idx = 0
        filtered_index_mapper = {}
        if self.split != "test":
            self.id2datum = {}
            for global_index in range(self.__len__()):
                index, question_index = self.index_mapper[global_index]
                qid = self.table["question_id"][index][question_index].as_py()
                answers = self.table["answers"][index][question_index].as_py()
                question_type = self.table["question_type"][index][
                    question_index
                ].as_py()
                # todo filter here?
                answer_type = self.table["answer_type"][index][question_index].as_py()
                if answer_filter is not None:
                    if answer_type != answer_filter or (
                        self.split == "val" and global_index > 35000
                    ):
                        # drop this sample from index mapper
                        del self.index_mapper[global_index]
                    else:
                        # change key index to crt_global_idx for consecutive index_mapper
                        filtered_index_mapper[crt_global_idx] = self.index_mapper[
                            global_index
                        ]
                        crt_global_idx += 1
                        continue

                self.id2datum[qid] = {
                    "answers": answers,
                    "question_type": question_type,
                    "answer_type": answer_type,
                }
        if answer_filter is not None:
            self.index_mapper = filtered_index_mapper

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["labels"][index][question_index].as_py()
            question_type = self.table["question_type"][index][question_index].as_py()
            answer_type = self.table["answer_type"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            question_type = list()
            answer_type = list()

        # todo overfit all yes for debugging
        return {
            "image": image_tensor,
            "text": text,
            "answers": answers,
            "labels": labels,
            "question_type": question_type,
            "answer_type": answer_type,
            "qid": qid,
        }

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)


if __name__ == "__main__":
    # mmsd = MMSDDataset(data_root="/root/autodl-tmp/mmsd/mmsd_raw_data", split="val")
    # a = mmsd[2]
    # a
    mmsd_loader = create_mmsd_loaders(
        data_root="/root/autodl-tmp/mmsd/mmsd_raw_data",
        batch_size=8,
        num_workers=2,
        n_shot=0,
    )

    for wav, text, frames, sarcasm in mmsd_loader[0]:
        print(sarcasm)
