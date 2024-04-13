import torch
import torch.nn.functional as F
import pdb

def bce_rescale_loss(config, logits_text, logits_visual, logits_iou, iou_mask_map, gt_maps, gt_times, word_label, word_mask, visual_content_emb, text_query_emb):
    B, D, T, _ = visual_content_emb.shape


    ## for AutoEncoder Teacher

    # regression loss
    reg_mask = (gt_maps[:,0:1,:T,None] >= 0.4) * (gt_maps[:,1:2,None,:] >= 0.4)
    gt_tmp = torch.cat((gt_maps[:,3:4,:T,None].repeat(1,1,1,T), gt_maps[:,4:5,None,:].repeat(1,1,T,1)), 1)
    loss_reg = (torch.abs(logits_iou[:,:2,:,:] - gt_tmp) * reg_mask).sum((2,3)) / reg_mask.sum((2,3))

    # iou loss
    idxs = torch.arange(T, device=logits_iou.device)
    s_e_idx = torch.cat((idxs[None,None,:T,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T,1)), 1)
    s_e_time = (s_e_idx + logits_iou[:,:2,:,:]).clone().detach()

    iou = torch.clamp(torch.min(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.max(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000000001) / torch.clamp(torch.max(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.min(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000001)

    temp = (s_e_time[:,0,:,:] < s_e_time[:,1,:,:]) * iou_mask_map[None,:,:]
    # iou[iou > 0.7] = 1.
    iou[iou < 0.5] = 0.
    loss_iou = (F.binary_cross_entropy_with_logits(logits_iou[:,2,:,:], iou, reduction='none') * temp * torch.pow(torch.sigmoid(logits_iou[:,2,:,:]) - iou, 2)).sum((1,2)) / temp.sum((1,2))
    # pdb.set_trace()
    ## for Contrastive Learning

    pos_sample_num = 1
    neg_sample_num = 120

    iou_mask = torch.clamp(torch.min(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.max(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000000001) / torch.clamp(torch.max(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.min(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000001)
    max_start_idx = iou_mask.reshape(B, -1).topk(k=pos_sample_num, dim=-1)[-1]//T
    max_end_idx = iou_mask.reshape(B, -1).topk(k=pos_sample_num, dim=-1)[-1]%T
    pos_iou_idx = torch.stack((max_start_idx, max_end_idx), dim=1).reshape(-1, 2)

    visual_pos_emb = visual_content_emb.permute(0,2,3, 1)[:, pos_iou_idx[:,0],pos_iou_idx[:,1], :].reshape(B, B, pos_sample_num, D)
    visual_pos_emb = torch.diagonal(visual_pos_emb, dim1=0, dim2=1).permute(2, 0, 1)

    min_start_idx = iou_mask.reshape(B, -1).topk(k=neg_sample_num, dim=-1, largest=False)[-1]//T
    min_end_idx = iou_mask.reshape(B, -1).topk(k=neg_sample_num, dim=-1, largest=False)[-1]%T
    neg_iou_idx = torch.stack((min_start_idx, min_end_idx), dim=1).reshape(-1, 2)

    visual_neg_emb = visual_content_emb.permute(0,2,3, 1)[:, neg_iou_idx[:,0],neg_iou_idx[:,1], :].reshape(B, B, neg_sample_num, D)
    visual_neg_emb_ivc = torch.diagonal(visual_neg_emb, dim1=0, dim2=1).permute(2, 0, 1)
    visual_neg_emb_cvc = torch.diagonal(visual_neg_emb, dim1=1, dim2=0).permute(2, 0, 1)

    text_query_emb = F.normalize(text_query_emb, p=2, dim=-1)
    visual_pos_emb = F.normalize(visual_pos_emb, p=2, dim=-1)
    visual_neg_emb_ivc = F.normalize(visual_neg_emb_ivc, p=2, dim=-1)
    visual_neg_emb_cvc = F.normalize(visual_neg_emb_cvc, p=2, dim=-1)

    temperature = 0.1
    score_pos = torch.bmm(text_query_emb.unsqueeze(dim=1), visual_pos_emb.permute(0, 2, 1))/temperature
    score_neg_ivc = torch.bmm(text_query_emb.unsqueeze(dim=1), visual_neg_emb_ivc.permute(0, 2, 1))/temperature
    score_neg_cvc = torch.bmm(text_query_emb.unsqueeze(dim=1), visual_neg_emb_cvc.permute(0, 2, 1))/temperature

    score_1 = score_pos.exp()

    score_2 = score_1 + (score_neg_ivc).exp().sum() + (score_neg_cvc).exp().sum()

    loss_cont = (-(torch.log(score_1) - torch.log(score_2))).mean()


    # token score loss
    joint_prob = torch.sigmoid(logits_visual[:,:3,:])
    gt_p = gt_maps[:,:3,:]
    loss_ts = F.binary_cross_entropy_with_logits(logits_visual[:,:3,:], gt_p, reduction='none') * (joint_prob-gt_p) * (joint_prob-gt_p)

    # reconstruction loss
    log_p = F.log_softmax(logits_text, -1)*word_mask.unsqueeze(2)
    grid = torch.arange(log_p.shape[-1], device=log_p.device).repeat(log_p.shape[0], log_p.shape[1], 1)
    text_loss = torch.sum(-log_p[grid==word_label.unsqueeze(2)]) / torch.clamp((word_mask.sum(1)>0).sum(), min=0.00000001)

    loss_value = config.W1*loss_ts.sum(-1).mean() + config.W2*loss_reg.mean() + config.W3*loss_iou.mean() + config.W4*text_loss + config.W5*loss_cont

    return loss_value, joint_prob, torch.sigmoid(logits_iou[:,2,:,:])*temp, s_e_time


def bce(logits_iou, iou_mask_map, gt_times, visual_content_emb):

    B, D, T, _ = visual_content_emb.shape

    # iou loss
    idxs = torch.arange(T, device=logits_iou.device)
    s_e_idx = torch.cat((idxs[None,None,:T,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T,1)), 1)
    s_e_time = (s_e_idx + logits_iou[:,:2,:,:]).clone().detach()

    iou = torch.clamp(torch.min(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.max(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000000001) / torch.clamp(torch.max(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.min(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000001)

    temp = (s_e_time[:,0,:,:] < s_e_time[:,1,:,:]) * iou_mask_map[None,:,:]
    # iou[iou > 0.7] = 1.
    iou[iou < 0.5] = 0.
    loss_iou = (F.binary_cross_entropy_with_logits(logits_iou[:,2,:,:], iou, reduction='none') * temp * torch.pow(torch.sigmoid(logits_iou[:,2,:,:]) - iou, 2)).sum((1,2)) / temp.sum((1,2))

    loss_value = 0.1 * loss_iou.mean()

    return loss_value


def simple_contrast(gt_times, pos_iou, neg_iou, visual_content_emb):
    B, D, T, _ = visual_content_emb.shape
    # iou loss
    idxs = torch.arange(T, device=pos_iou.device)
    s_e_idx = torch.cat((idxs[None,None,:T,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T,1)), 1)
    s_e_time = (s_e_idx + pos_iou[:,:2,:,:]).clone().detach()
    iou = torch.clamp(torch.min(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.max(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000000001) / torch.clamp(torch.max(gt_times[:,1][:,None,None], s_e_time[:,1,:,:]) - torch.min(gt_times[:,0][:,None,None], s_e_time[:,0,:,:]), min=0.0000001)
    pos_iou = pos_iou[:,2,:,:]
    neg_iou = neg_iou[:,2,:,:]

    batch_size = pos_iou.shape[0]
    map_h_ori_reshape = iou.reshape(batch_size, -1)
    map_h_pos_reshape = pos_iou.reshape(batch_size, -1)
    map_h_neg_reshape = neg_iou.reshape(batch_size, -1)
    pos = torch.cosine_similarity(map_h_ori_reshape, map_h_pos_reshape, dim=1)
    neg = torch.cosine_similarity(map_h_ori_reshape, map_h_neg_reshape, dim=1)
    logit = torch.stack((pos, neg), 1) # [b, 2]
    contras_loss = -F.log_softmax(logit, 1)
    loss_value = contras_loss[:, 0].mean()
    # pdb.set_trace()
    return loss_value * 0.1