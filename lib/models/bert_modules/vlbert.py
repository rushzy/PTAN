import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_linguistic_bert import VisualLinguisticBert
# from .unimodal_bert import UniModalBert
import pdb

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class TLocVLBERT(nn.Module):
    def __init__(self, dataset, config):

        super(TLocVLBERT, self).__init__()

        self.config = config

        language_pretrained_model_path = None
        if config.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.BERT_PRETRAINED,
                                                                      config.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.BERT_MODEL_NAME):
            weight_path = os.path.join(config.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        if dataset ==  "ActivityNet":
            iou_mask_map = torch.zeros(33,33).float()
            for i in range(0,32,1):
                iou_mask_map[i,i+1:min(i+17,33)] = 1.
            for i in range(0,32-16,2):
                iou_mask_map[i,range(18+i,33,2)] = 1.
        elif dataset ==  "Charades":
            iou_mask_map = torch.zeros(33,33).float()
            for i in range(0,32,1):
                iou_mask_map[i,i+1:min(i+17,33)] = 1.
            for i in range(0,32-16,2):
                iou_mask_map[i,range(18+i,33,2)] = 1.
        else:
            print('DATASET ERROR')
            exit()

        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vlbert = VisualLinguisticBert(dataset, config, language_pretrained_model_path=language_pretrained_model_path)


        dim = config.hidden_size
        if config.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.vocab_size)
            )
            self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, dim*3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
            )
            self.final_mlp_3 = torch.nn.Sequential(
                torch.nn.Linear(dim*3, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 3)
            )

            self.final_mlp_s = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
            self.final_mlp_e = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
            self.final_mlp_c = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )

        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        self.project_txt = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.PROJECTOR_LATENT_SIZE)
            )

        self.project_vid = torch.nn.Sequential(
                torch.nn.Linear(dim*3, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.PROJECTOR_LATENT_SIZE)
            )


        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_3.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.project_txt.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.project_vid.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def fix_params(self):
        pass


    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats):

        hidden_states_text, hidden_states_object, _, _ = self.vlbert(text_input_feats, text_mask, word_mask, object_visual_feats, output_all_encoded_layers=False)
        # pdb.set_trace()
        tmp_state = hidden_states_object
        logits_text = self.final_mlp(hidden_states_text)
        hidden_states_object = self.final_mlp_2(hidden_states_object)
        hidden_s, hidden_e, hidden_c = torch.split(hidden_states_object, self.config.hidden_size, dim=-1)


        T = hidden_states_object.size(1)
        s_idx = torch.arange(T, device=hidden_states_object.device)
        e_idx = torch.arange(T, device=hidden_states_object.device)
        c_point = hidden_c[:,(0.5*(s_idx[:,None] + e_idx[None,:])).long().flatten(),:].view(hidden_c.size(0),T,T,hidden_c.size(-1))
        s_c_e_points = torch.cat((hidden_s[:,:,None,:].repeat(1,1,T,1), c_point, hidden_e[:,None,:,:].repeat(1,T,1,1)), -1)
        # srart, end, iou
        logits_iou = self.final_mlp_3(s_c_e_points).permute(0,3,1,2).contiguous()
        # pdb.set_trace()

        visual_point = tmp_state[:,(0.5*(s_idx[:,None] + e_idx[None,:])).long().flatten(),:].view(tmp_state.size(0),T,T,tmp_state.size(-1))
        visual_point = torch.cat((hidden_s[:,:,None,:].repeat(1,1,T,1), visual_point, hidden_e[:,None,:,:].repeat(1,T,1,1)), -1)
        text_point = hidden_states_text[:,0,:]

        visual_content_emb = self.project_vid(visual_point).permute(0,3,1,2).contiguous()
        text_query_emb = self.project_txt(text_point).contiguous()

        logits_visual = torch.cat((self.final_mlp_s(hidden_s), self.final_mlp_e(hidden_e), self.final_mlp_c(hidden_c)), -1)

        return logits_text, logits_visual, logits_iou, self.iou_mask_map.clone().detach(), visual_content_emb, text_query_emb
