import torch
import torch.nn as nn
from lib.core.config import config

def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_word_label = [b['word_label'] for b in batch]
    batch_word_mask = [b['word_mask'] for b in batch]
    batch_gt_times = [b['gt_times'].unsqueeze(0) for b in batch]

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 5, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][:,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
        'batch_word_label': nn.utils.rnn.pad_sequence(batch_word_label, batch_first=True).long(),
        'batch_word_mask': nn.utils.rnn.pad_sequence(batch_word_mask, batch_first=True).float(),
        'batch_gt_times': torch.cat(batch_gt_times, 0),
        'is_train': False
    }

    return batch_data

def train_collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_word_label = [b['word_label'] for b in batch]
    batch_word_mask = [b['word_mask'] for b in batch]
    batch_gt_times = [b['gt_times'].unsqueeze(0) for b in batch]
    pos_vid_feat = [b['pos_vid_feat'] for b in batch]
    neg_vid_feat = [b['neg_vid_feat'] for b in batch]
    pos_sent_feat = [b['pos_sent_feat'] for b in batch]
    neg_sent_feat = [b['neg_sent_feat'] for b in batch]
    pos_weight = [b['pos_weight'] for b in batch]
    neg_weight = [b['neg_weight'] for b in batch]


    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 5, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][:,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
        'batch_word_label': nn.utils.rnn.pad_sequence(batch_word_label, batch_first=True).long(),
        'batch_word_mask': nn.utils.rnn.pad_sequence(batch_word_mask, batch_first=True).float(),
        'batch_gt_times': torch.cat(batch_gt_times, 0),
        'pos_vid_feat': nn.utils.rnn.pad_sequence(pos_vid_feat, batch_first=True).float(),
        'neg_vid_feat': nn.utils.rnn.pad_sequence(neg_vid_feat, batch_first=True).float(),
        'pos_sent_feat': nn.utils.rnn.pad_sequence(pos_sent_feat, batch_first=True),
        'neg_sent_feat': nn.utils.rnn.pad_sequence(neg_sent_feat, batch_first=True),
        'pos_weight': torch.tensor(pos_weight),
        'neg_weight': torch.tensor(neg_weight),
        'is_train': True
    }

    return batch_data

def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

from lib.datasets.charades import Charades
from lib.datasets.activitynet import ActivityNet