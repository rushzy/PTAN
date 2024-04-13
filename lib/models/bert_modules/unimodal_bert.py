from __future__ import division
import torch
import torch.nn as nn
from .intra_modeling import BertLayerNorm, UniBertEncoder, ACT2FN
import numpy as np
import math

# todo: add this to config
# NUM_SPECIAL_WORDS = 1000

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=116):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class UniModalBert(BaseModel):
    def __init__(self, dataset, config, language_pretrained_model_path=None, modal_input = "TXT"):
        super(UniModalBert, self).__init__(config)

        self.config = config

        if dataset == "ActivityNet":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=116)
        elif dataset == "TACoS":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        elif dataset == "Charades":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        elif dataset == "MedVidQA":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        elif dataset == "YouMakeUp":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        else:
            print('DATASET ERROR')
            exit()


        # embeddings
        # self.mask_embeddings = nn.Embedding(1, config.hidden_size) 
        # self.word_mapping = nn.Linear(300, config.hidden_size)    # 300 is the dim of glove vector
        # self.embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.mask_embeddings = nn.Embedding(1, config.hidden_size) 
        self.word_mapping = nn.Linear(300, config.hidden_size)    # 300 is the dim of glove vector
        self.text_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.text_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

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

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.visual_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.visual_size, config.hidden_size)
        if config.visual_ln:
            self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.encoder = UniBertEncoder(config, modal_input)

        self.modality = modal_input

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            print('load language pretrained model')
            self.load_language_pretrained_model(language_pretrained_model_path)

    def forward(self,
                text_input_feats,
                text_mask,
                word_mask,
                object_visual_embeddings,
                output_all_encoded_layers=False,
                output_attention_probs=False):

        # get seamless concatenate embeddings and mask
        text_embeddings, visual_embeddings = self.embedding(text_input_feats,
                                                            text_mask, word_mask,
                                                            object_visual_embeddings)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = text_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        # extended_attention_mask = 1.0 - extended_attention_mask
        # extended_attention_mask[extended_attention_mask != 0] = float('-inf')

        input_embeddings = text_embeddings if self.modality == "TXT" else visual_embeddings

        if output_attention_probs:
            encoded_layers, attention_probs = self.encoder(input_embeddings,
                                                            extended_attention_mask,
                                                            output_all_encoded_layers=output_all_encoded_layers,
                                                            output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.encoder(input_embeddings,
                                           extended_attention_mask,
                                           output_all_encoded_layers=output_all_encoded_layers,
                                           output_attention_probs=output_attention_probs)
            
        # sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        if output_all_encoded_layers:
            encoded_layers_text = []
            for encoded_layer in encoded_layers:
                encoded_layers_text.append(encoded_layer[0])
            if output_attention_probs:
                attention_probs_text = []
                for attention_prob in attention_probs:
                    attention_probs_text.append(attention_prob[0])
                return encoded_layers_text, attention_probs_text
            else:
                return encoded_layers_text
        else:
            encoded_layers = encoded_layers[-1]
            if output_attention_probs:
                attention_probs = attention_probs[-1]
                return encoded_layers[0], attention_probs[0]
            else:
                return encoded_layers[0]

    def embedding(self,
                  text_input_feats,
                  text_mask,
                  word_mask,
                  object_visual_embeddings):

        text_linguistic_embedding = self.word_mapping(text_input_feats)
        text_input_feats_temp = text_input_feats.clone()
        mask_word_mean = text_mask
        if self.training:
            text_input_feats_temp[word_mask>0] = 0
            mask_word_mean = text_mask * (1. - word_mask)
            _zero_id = torch.zeros(text_linguistic_embedding.shape[:2], dtype=torch.long, device=text_linguistic_embedding.device)
            text_linguistic_embedding[word_mask>0] = self.mask_embeddings(_zero_id)[word_mask>0]

        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        if self.config.visual_ln:
            object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)

        embeddings = torch.cat([object_visual_embeddings, text_linguistic_embedding], dim=1)
        embeddings = self.postion_encoding(embeddings)
        visual_embeddings, text_embeddings = torch.split(embeddings, [object_visual_embeddings.size(1),text_linguistic_embedding.size(1)], 1)

        text_embeddings = self.text_embedding_LayerNorm(text_embeddings)
        text_embeddings = self.text_embedding_dropout(text_embeddings)

        visual_embeddings = self.visual_embedding_LayerNorm(visual_embeddings)
        visual_embeddings = self.visual_embedding_dropout(visual_embeddings)

        return text_embeddings, visual_embeddings

    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        unexpected_keys = []
        for k, v in pretrained_state_dict.items():
            if k.startswith('bert.'):
                k = k[len('bert.'):]
            elif k.startswith('roberta.'):
                k = k[len('roberta.'):]
            else:
                unexpected_keys.append(k)
                continue
            if 'gamma' in k:
                k = k.replace('gamma', 'weight')
            if 'beta' in k:
                k = k.replace('beta', 'bias')
            if k.startswith('encoder.'):
                k_ = k[len('encoder.'):]
                if k_ in self.encoder.state_dict():
                    encoder_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            elif k.startswith('embeddings.'):
                k_ = k[len('embeddings.'):]
                if k_ == 'word_embeddings.weight':
                    self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                            device=self.word_embeddings.weight.data.device)
                elif k_ == 'position_embeddings.weight':
                    self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                device=self.position_embeddings.weight.data.device)
                elif k_ == 'token_type_embeddings.weight':
                    self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                        dtype=self.token_type_embeddings.weight.data.dtype,
                        device=self.token_type_embeddings.weight.data.device)
                    if v.size(0) == 1:
                        # Todo: roberta token type embedding
                        self.token_type_embeddings.weight.data[1] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        self.token_type_embeddings.weight.data[2] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)

                elif k_.startswith('LayerNorm.'):
                    k__ = k_[len('LayerNorm.'):]
                    if k__ in self.embedding_LayerNorm.state_dict():
                        embedding_ln_pretrained_state_dict[k__] = v
                    else:
                        unexpected_keys.append(k)
                else:
                    unexpected_keys.append(k)
            elif self.config.with_pooler and k.startswith('pooler.'):
                k_ = k[len('pooler.'):]
                if k_ in self.pooler.state_dict():
                    pooler_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            else:
                unexpected_keys.append(k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)

