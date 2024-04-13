""" Dataset loader for the ActivityNet Captions dataset """
import os
import json
import random
from collections import OrderedDict
import numpy as np

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import tqdm
import nltk

from . import average_to_fixed_length
from lib.core.eval import iou
from lib.core.config import config

class ActivityNet(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"](cache="./data/ActivityNet")
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(ActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.json_dir = config.JSON_DIR
        self.split = split

        self.tag2bit_mask = {'N': 0b00001, 'V':0b00010, 'J': 0b00100, 'R':0b01000, 'O':0b10000}
        self.tag2weight = {'N': 1, 'V': 1, 'J': 0.6, 'R': 0.6, 'O': 0}

        with open('./data/ActivityNet/words_vocab_activitynet_14753.json', 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i

        with open(os.path.join(self.json_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        self.tmp_annotations = annotations
        anno_pairs = []
        max_sent_len = 0
        for vid, video_anno in annotations.items():
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    sentence = sentence.replace(',',' ').replace('/',' ').replace('\"',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('&',' ').replace('?',' ').replace('!',' ').replace('(',' ').replace(')',' ')
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0],0),min(timestamp[1],duration)],
                            'description':sentence,
                        }
                    )
                    if len(sentence.split()) > max_sent_len:
                        max_sent_len = len(sentence.split())

        self.annotations = anno_pairs
        print('max_sent_len', max_sent_len)

        self.sent_tags, self.neg_tag_mask, self.pos_tag_mask = self.get_sent_tags()

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_label = [self.stoi.get(w.lower(), 10727) for w in sentence.split()]
        range_i = range(len(word_label))
        # if np.random.uniform(0,1)<0.8:
        word_mask = [1. if np.random.uniform(0,1)<0.15 else 0. for _ in range_i]
        if np.sum(word_mask) == 0.:
            mask_i = np.random.choice(range_i)
            word_mask[mask_i] = 1.
        if np.sum(word_mask) == len(word_mask):
            unmask_i = np.random.choice(range_i)
            word_mask[unmask_i] = 0.

        word_label = torch.tensor(word_label, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.float)

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)


        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
            # torch.arange(0,)

        map_gt = np.zeros((5, num_clips+1), dtype=np.float32)

        clip_duration = duration/num_clips
        gt_s = gt_s_time/clip_duration
        gt_e = gt_e_time/clip_duration
        gt_length = gt_e - gt_s
        gt_center = (gt_e + gt_s) / 2.
        map_gt[0, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
        map_gt[0, map_gt[0, :]>=0.6] = 1.
        map_gt[0, map_gt[0, :]<0.1353] = 0.
        map_gt[1, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
        map_gt[1, map_gt[1, :]>=0.6] = 1.
        map_gt[1, map_gt[1, :]<0.1353] = 0.
        # map_gt[2, gt_s_idx:gt_e_idx] = 1.
        map_gt[2, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
        map_gt[2, map_gt[2, :]>=0.78] = 1.
        map_gt[2, map_gt[2, :]<0.0625] = 0.
        map_gt[3, :] = gt_s - np.arange(num_clips+1)
        map_gt[4, :] = gt_e - np.arange(num_clips+1)
        if (map_gt[0, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        if map_gt[2, :].sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
            idx = np.argmax(p)
            map_gt[2, idx] = 1.

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0],),
            'map_gt': torch.from_numpy(map_gt),
            'word_label': word_label,
            'word_mask': word_mask,
            'gt_times': torch.from_numpy(np.array([gt_s, gt_e], dtype=np.float32))
        }

        if self.split != 'train':
            return item

        s_flag = random.random() < 0.5
        v_flag = not s_flag

        if s_flag:
            neg_sent_mask, neg_sent_weight = self.get_sent_sample(self.sent_tags[index], self.neg_tag_mask)
            assert word_vectors.size(0) == neg_sent_mask.size(0), f"{sentence} \n {word_idxs} \n {neg_sent_mask}"
            neg_sent_feat = word_vectors * neg_sent_mask[..., None]
        else:
            neg_sent_feat = word_vectors
            neg_sent_weight = 0
        if v_flag:
            neg_video_mask, neg_video_weight = self.get_video_sample(visual_input, [gt_s, gt_e], duration, mode='n')
            neg_video_feat = visual_input * neg_video_mask[..., None]
        else:
            neg_video_feat = visual_input
            neg_video_weight = 0

        s_flag = random.random() < 0.5
        v_flag = not s_flag

        if s_flag:
            pos_sent_mask, pos_sent_weight = self.get_sent_sample(self.sent_tags[index], self.pos_tag_mask)
            assert word_vectors.size(0) == pos_sent_mask.size(0), f"{sentence} \n {word_idxs} \n {pos_sent_mask}"
            pos_sent_feat = word_vectors * pos_sent_mask[..., None]
        else:
            pos_sent_feat = word_vectors
            pos_sent_weight = 0
        if v_flag:
            pos_video_mask, pos_video_weight = self.get_video_sample(visual_input, [gt_s, gt_e], duration, mode='p')
            pos_video_feat = visual_input * pos_video_mask[..., None]
        else:
            pos_video_feat = visual_input
            pos_video_weight = 0

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0],),
            'map_gt': torch.from_numpy(map_gt),
            'word_label': word_label,
            'word_mask': word_mask,
            'gt_times': torch.from_numpy(np.array([gt_s, gt_e], dtype=np.float32)),
            'pos_vid_feat': pos_video_feat,
            'neg_vid_feat': neg_video_feat,
            'pos_sent_feat': pos_sent_feat,
            'neg_sent_feat': neg_sent_feat,
            'pos_weight': pos_sent_weight + pos_video_weight,
            'neg_weight': neg_sent_weight + neg_video_weight,
        }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask


    def get_sent_tags(self):
        if self.split != 'train':
            return None, None, None

        MASK_WORD_TYPE = 'RandM_NVJR'
        neg_tag_mask = 0b00000000
        for tag_mask in MASK_WORD_TYPE.split('_', 1)[1]:
            if tag_mask in ['N', 'V', 'J', 'R', 'O']:
                neg_tag_mask |= self.tag2bit_mask[tag_mask]
            else:
                raise NotImplementedError

        MASK_WORD_TYPE = 'RandM_O'
        pos_tag_mask = 0b00000000
        for tag_mask in MASK_WORD_TYPE.split('_', 1)[1]:
            if tag_mask in ['N', 'V', 'J', 'R', 'O']:
                pos_tag_mask |= self.tag2bit_mask[tag_mask]
            else:
                raise NotImplementedError

        sent_tags = []
        # for vid, annotation in tqdm.tqdm(self.tmp_annotations.items(), desc='Loading sent tags'):
        for annotation in tqdm.tqdm(self.annotations, desc='Loading sent tags'):
            sentences = [annotation['description']]
            for sent in sentences:
                # sent = sent.replace(',',' ').replace('/',' ').replace('\"',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('&',' ').replace('?',' ').replace('!',' ').replace('(',' ').replace(')',' ')
                # token = tokenizer(sent)   #ignore ','
                token = sent.split()
                sent_tag, sent_weight = [], []
                for tag in nltk.pos_tag(token):
                    if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                        sent_tag.append(self.tag2bit_mask['N'])
                        sent_weight.append(self.tag2weight['N'])
                    elif tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        sent_tag.append(self.tag2bit_mask['V'])
                        sent_weight.append(self.tag2weight['V'])
                    elif tag[1] in ['JJ', 'JJR', 'JJS']:
                        sent_tag.append(self.tag2bit_mask['J'])
                        sent_weight.append(self.tag2weight['J'])
                    elif tag[1] in ['RB', 'RBR', 'RBS', 'RP']:
                        sent_tag.append(self.tag2bit_mask['R'])
                        sent_weight.append(self.tag2weight['R'])
                    else:
                        sent_tag.append(self.tag2bit_mask['O'])
                        sent_weight.append(self.tag2weight['O'])
                sent_tags.append((torch.tensor(sent_tag, dtype=torch.int32),
                                    torch.tensor(sent_weight)))
        return sent_tags, neg_tag_mask, pos_tag_mask

    def get_video_sample(self, video_feat, gt_times, duration, mode='n'):
        vis_len = video_feat.size(0)
        vis_mask = np.ones(vis_len)
        st_idx = max(round(gt_times[0] * vis_len / duration), 0)
        ed_idx = min(round(gt_times[1] * vis_len / duration), vis_len - 1)
        gt_len = max(1, ed_idx - st_idx)

        if(mode == 'n'):
            # 负样本
            # ground_truth内的帧掩掉40%
            prop_mask = np.random.rand(vis_len)
            prop_mask[st_idx:ed_idx] = 1
            prop = np.exp(prop_mask) / np.sum(np.exp(prop_mask))
            choices = np.random.choice(np.arange(vis_len), int(vis_len * 0.5), replace=False, p=prop)
            vis_mask[choices] = 0

            vis_mask[st_idx:ed_idx] = 1

            prop_mask = np.random.rand(gt_len)
            prop = np.exp(prop_mask) / np.sum(np.exp(prop_mask))
            choices = np.random.choice(np.arange(gt_len), int(gt_len * 0.4), replace=False, p=prop)
            vis_mask[st_idx + choices] = 0
        else:
            # positive样本
            num_zeros = vis_len // 4
            prop_mask = np.random.rand(vis_len)
            prop_mask[st_idx:ed_idx] = 1
            sorted_idx = np.argsort(prop_mask)
            idx2zero = sorted_idx[:vis_len//2]
            np.random.shuffle(idx2zero)
            idx2zero = idx2zero[:num_zeros]
            vis_mask[idx2zero] = 0

        sample_weight = vis_mask[st_idx:ed_idx].sum()
        weight = sample_weight / gt_len
        return torch.from_numpy(vis_mask), weight

    def get_sent_sample(self, sent_tag, tag_mask):
        """

        : param sent_tag: Tag and weight of sentence, tuple(list[L], list[L])
        : param tag_mask: Bit tag mask, int
        : return: (sample mask, weight), where 0 reprent to mask, tuple(list[L], int)
        """
        op1 = 'Rand'
        op2 = 'M'
        sent_tag, sent_weight = sent_tag
        sample_mask = (sent_tag & tag_mask) != 0
        if op1 == 'Rand':
            tmp_mask = torch.rand_like(sample_mask, dtype=torch.float32) < 0.5
            sample_idx = (sample_mask * tmp_mask).nonzero()
        sample_mask = torch.ones_like(sample_mask)
        sample_mask[sample_idx] = False

        batch_weight = 1
        if True:
            sample_weight = sent_weight[~sample_mask].sum()
            total_weight = sent_weight.sum()
            batch_weight = sample_weight / total_weight

        assert op2 == 'M'
        return sample_mask, batch_weight