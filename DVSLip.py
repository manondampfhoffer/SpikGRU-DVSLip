"""
Dataset pre-processing
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as trans

class DVSLipDataset(Dataset):
    def __init__(self, data_root, label_dct, train=True, augment_spatial=False, augment_temporal=False, T=90, Tnbmask=6, Tmaxmasklength=18):
        self.filenames = []
        self.labels = []
        self.augment_spatial = augment_spatial
        self.augment_temporal = augment_temporal
        self.train = train
        self.T = T

        if self.train:
            self.base_transform = trans.Compose([trans.CenterCrop(96), trans.RandomCrop(88), trans.RandomHorizontalFlip(0.5)])
        else:
            self.base_transform = trans.CenterCrop(88)
        if self.augment_spatial:
            self.cutout = Cutout(n_holes=4, max_length=20)
            self.zoom = Zoom(max_scale=26)
        if self.augment_temporal:
            # self.timemask = Masking(nb_mask=6, max_mask_length=self.T // 5)
            self.timemask = Masking(nb_mask=Tnbmask, max_mask_length=Tmaxmasklength)

        for root, dirs, files in os.walk(data_root):
            for filename in files:
                word = root.split("/")[-1]
                label = label_dct.get(word)
                if label is None:
                    print("ignored word: %s"%word)
                    break
                partial_path = '/'.join([word, filename])
                full_name = os.path.join(root, filename)
                self.filenames.append(full_name)
                self.labels.append(label)
                
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        X = 128; Y = 128
        # load sample
        sample = np.array(np.load(filename).tolist()) # list of elements of size : (t, x, y, p)

        # PRE-PROCESSING
        time_step = 4e4 / (self.T / 30)
        ts = (np.round(sample[:,0] / time_step).astype(np.int))
        # remove events >= T
        restrict_idx = (ts < self.T)
        ts = ts[restrict_idx]
        xs = sample[restrict_idx,1]
        ys = sample[restrict_idx,2]
        polarity = sample[restrict_idx, 3]
        # compute polarity (p=0 corresponds to polarity -1 and 1 corresponds to polarity 1)
        p_idx_neg = (polarity == 0)
        polarity[p_idx_neg] = -1 

        # separate positive and negative events in 2 channels
        p_idx_pos = np.logical_not(p_idx_neg)
        ts_pos = ts[p_idx_pos]
        xs_pos = xs[p_idx_pos]
        ys_pos = ys[p_idx_pos]
        polarity_pos = polarity[p_idx_pos]
        ts_neg = ts[p_idx_neg]
        xs_neg = xs[p_idx_neg]
        ys_neg = ys[p_idx_neg]
        polarity_neg = polarity[p_idx_neg]
        coo_pos = [[] for i in range(3)]
        coo_pos[0].extend(ts_pos)
        coo_pos[1].extend(xs_pos)
        coo_pos[2].extend(ys_pos)
        i_pos = torch.LongTensor(coo_pos)
        v_pos = torch.FloatTensor(polarity_pos)
        item_pos = torch.sparse.FloatTensor(i_pos, v_pos, torch.Size([self.T, X, Y])).to_dense()
        coo_neg = [[] for i in range(3)]
        coo_neg[0].extend(ts_neg)
        coo_neg[1].extend(xs_neg)
        coo_neg[2].extend(ys_neg)
        i_neg = torch.LongTensor(coo_neg)
        v_neg = torch.FloatTensor(polarity_neg)
        item_neg = torch.sparse.FloatTensor(i_neg, v_neg, torch.Size([self.T, X, Y])).to_dense()
        item = torch.stack((item_pos, item_neg))
        item = item.transpose(0,1) # output: (T, Cin, X, Y)
        
        # to put in pytorch order height, width (for horizontalflip) - apply before transform
        item = item.permute(0,1,3,2) # output: (T, Cin, Y, X)

        # DATA AUGMENTATION
        item = self.base_transform(item)
        if self.augment_spatial:
            item = self.cutout(item)
            item = self.zoom(item)
        if self.augment_temporal:
            item = self.timemask(item)
            
        label = self.labels[idx]
        return item, label


class Masking:
    """ Randomly mask several consecutive frames of the sample (temporal data augmentation)
        nb: in case several masks are used, masks can overlap
    Args:
        nb_mask (int): number of masks to apply
        max_mask_length (int): max size of a mask
    """
    def __init__(self, nb_mask, max_mask_length):
        self.nb_mask = nb_mask
        self.max_mask_length = max_mask_length

    def __call__(self, item):
        # item : (T, Cin, Y, X)
        for i in range(self.nb_mask):
            if self.max_mask_length == 1:
                mask_length_T = 1
            else: 
                mask_length_T = np.random.randint(0, self.max_mask_length)
            start_T = np.random.randint(0, item.size(0) - mask_length_T)
            item[start_T:start_T + mask_length_T, :, :, :] = 0
        return item


class Zoom:
    """ Randomly zoom-in or zoom-out the given sample (spatial data augmentation)
    Args:
        max_scale (int): max number of pixels to add (or remove) from the image outline
    """
    def __init__(self, max_scale):
        self.max_scale = max_scale

    def __call__(self, item):
        # item : (T, Cin, Y, X)
        p = np.random.random()
        if p < 0.5: ## zoom in:
            scale = np.random.randint(88 - self.max_scale, 88)
            zoom = trans.Compose([trans.CenterCrop(scale), trans.Resize(88)])
        else: ## zoom out:
            scale = np.random.randint(0, self.max_scale)
            zoom = trans.Compose([trans.Pad(scale), trans.Resize(88)])
        item = zoom(item)
        return item


class Cutout(object):
    ## from https://github.com/Intelligent-Computing-Lab-Yale/NDA_SNN/blob/d355bcb813c6e00c162a20ae7fd5c817b7014c02/functions/data_loaders.py
    """Randomly mask out one or more patches from an image (spatial data augmentation)
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, max_length):
        self.n_holes = n_holes
        self.max_length = max_length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        for i in range(self.n_holes):
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            length = np.random.randint(1, self.max_length)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        return img


def get_training_words():
    classes = [
        "accused",
        "action",
        "allow",
        "allowed",
        "america",
        "american",
        "another",
        "around",
        "attacks",
        "banks",
        "become",
        "being",
        "benefit",
        "benefits",
        "between",
        "billion",
        "called",
        "capital",
        "challenge",
        "change",
        "chief",
        "couple",
        "court",
        "death",
        "described",
        "difference",
        "different",
        "during",
        "economic",
        "education",
        "election",
        "england",
        "evening",
        "everything",
        "exactly",
        "general",
        "germany",
        "giving",
        "ground",
        "happen",
        "happened",
        "having",
        "heavy",
        "house",
        "hundreds",
        "immigration",
        "judge",
        "labour",
        "leaders",
        "legal",
        "little",
        "london",
        "majority",
        "meeting",
        "military",
        "million",
        "minutes",
        "missing",
        "needs",
        "number",
        "numbers",
        "paying",
        "perhaps",
        "point",
        "potential",
        "press",
        "price",
        "question",
        "really",
        "right",
        "russia",
        "russian",
        "saying",
        "security",
        "several",
        "should",
        "significant",
        "spend",
        "spent",
        "started",
        "still",
        "support",
        "syria",
        "syrian",
        "taken",
        "taking",
        "terms",
        "these",
        "thing",
        "think",
        "times",
        "tomorrow",
        "under",
        "warning",
        "water",
        "welcome",
        "words",
        "worst",
        "years",
        "young",
    ]
    return classes