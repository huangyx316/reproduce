import torch
import clip
from PIL import Image
import json
import cv2
import numpy as np
from tqdm import tqdm
import math
from math import log
from torch.nn.utils.rnn import pad_sequence
import sys
import time
import os
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from itertools import chain
from transformers import AutoTokenizer, AutoModel

def compute_correlation_uniquehuman(pred, all_human_scores):
    num_workers = 3
    import scipy.stats

    pred = np.around(pred, decimals=4)

    spearman = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.spearmanr(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        spearman += tmp
    spearman /= num_workers
    spearman = np.around(spearman, decimals=4)

    kendalltau = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.kendalltau(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        kendalltau += tmp
    kendalltau /= num_workers
    kendalltau = np.around(kendalltau, decimals=4)

    print('kendall: {}, spear: {}'.format(kendalltau, spearman))
    return kendalltau, spearman

def normalize_matrix(A):
    assert len(A.shape) == 2
    A_norm = torch.linalg.norm(A, dim=-1, keepdim=True)
    return A / A_norm

def encode_video(video_file, preprocess, model, batch_size, device):

    cv_start_time = time.perf_counter()
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    count = 0
    ret = True
    
    while (count < frameCount and ret):
        ret, frame = cap.read()
        if not ret:  # if file is empty break loop
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        count += 1
    
    cv_end_time = time.perf_counter()
    time_diff = cv_end_time-cv_start_time
    # print(f"cv done in {time_diff:.2f} seconds")

    image_embed_start_time = time.perf_counter()
    image_input = torch.tensor(np.stack(images)).to(device)
    image_features_list = []
    # bs = 256
    with torch.no_grad():
        n_inter = math.ceil(len(image_input)/batch_size)
        for i in range(n_inter):
            image_features = model.encode_image(image_input[i*batch_size: (i+1)*batch_size]).float()
            image_features_list.append(image_features)
    image_features = torch.cat(image_features_list, dim=0)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    cap.release()

    vid_feature = normalize_matrix(torch.mean(image_features, dim=0, keepdim=True)).squeeze()

    image_embed_end_time = time.perf_counter()
    time_diff = image_embed_end_time - image_embed_start_time
    # print(f"image embed done in {time_diff:.2f} seconds")

    return image_features, vid_feature


#原始处理candidate text函数，需要将其改成处理code candidate text
# def encode_text(vid_caps, model, tokenizer, idf_dict, device):
#     # text_input = tokenizer(vid_caps).to(device=device)
#     # print(text_input)
#     # print(text_input.cpu())
#     # print(type(text_input.cpu()))
#     with torch.no_grad():
#         #此处是调用模型进行encode处理
        
#         # text_features = model.encode_text(text_input, local=True).float()
        
#         tokenizer1 = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
#         model1 = AutoModel.from_pretrained('microsoft/unixcoder-base')
#         # text_input1 = tokenizer1.encode_plus(vid_caps,add_special_tokens=True,padding="max_length",
#         #                                     max_length=128,return_attention_mask=True,
#         #                                     return_token_type_ids=False,return_overflowing_tokens=False,
#         #                                     return_special_tokens_mask=False)
        
#         text_input = tokenizer1.encode_plus(vid_caps, return_tensors='pt')
        
       
#         text_features = model1(**text_input)[0]
#         #mask= text_input['attention_mask']
#     text_features /= text_features.norm(dim=-1, keepdim=True)
    
#     # For special tokens, use [SOS] and [EOS]
#     txt_len = text_input['input_ids'].argmax(dim=-1)
#     mask = torch.zeros_like(text_input['input_ids'])
    
#     for i in range(len(mask)):
#         mask[i][0:txt_len[i]+1] = 1
    
#     # For special tokens, only use [EOS]
#     # txt_len = text_input.argmax(dim=-1)
#     # mask = torch.zeros_like(text_input)
#     # for i in range(len(mask)):
#     #     mask[i][1:txt_len[i]+1] = 1

#     # # For special tokens, don't use [SOS] and [EOS]
#     # txt_len = text_input.argmax(dim=-1)
#     # mask = torch.zeros_like(text_input)
#     # for i in range(len(mask)):
#     #     mask[i][1:txt_len[i]] = 1
    
#     #idf_weights = torch.tensor([[idf_dict[int(i)] for i in a] for a in text_input.cpu()])
#     idf_weights = torch.tensor([[idf_dict[int(i)] for i in a] for a in text_input['input_ids'].cpu()])
#     # print("原始方法输出：")
#     # print(text_features)
#     # print(len(text_features[0]))
#     # print(mask)
#     # print(idf_weights)
#     # print("-----------")
#     print("修改代码后方法输出：")
#     print(text_features)
#     print(mask)
#     print(idf_weights)
#     return text_features, mask, idf_weights

#备份未修改代码
def encode_text(vid_caps, model, tokenizer, idf_dict, device):
    
    text_input = tokenizer(vid_caps).to(device=device)

    print("test_input的一些信息")
    print(text_input)
    print(type(text_input))
    print(len(text_input))
    print(text_input.shape)

    with torch.no_grad():
        text_features = model.encode_text(text_input, local=True).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # For special tokens, use [SOS] and [EOS]
    txt_len = text_input.argmax(dim=-1)
    mask = torch.zeros_like(text_input)
    for i in range(len(mask)):
        mask[i][0:txt_len[i]+1] = 1
    
    # For special tokens, only use [EOS]
    # txt_len = text_input.argmax(dim=-1)
    # mask = torch.zeros_like(text_input)
    # for i in range(len(mask)):
    #     mask[i][1:txt_len[i]+1] = 1

    # # For special tokens, don't use [SOS] and [EOS]
    # txt_len = text_input.argmax(dim=-1)
    # mask = torch.zeros_like(text_input)
    # for i in range(len(mask)):
    #     mask[i][1:txt_len[i]] = 1
    
    idf_weights = torch.tensor([[idf_dict[int(i)] for i in a] for a in text_input.cpu()])

    print("文本处理之后的一些信息：")
    print(type(text_features))
    print(text_features)
    print(text_features.shape)

    print("关于mask的一些信息")
    print(type(mask))
    print(mask)
    print(mask.shape)

    print("关于idf权重的一些信息：")
    print(type(idf_weights))
    print(idf_weights)
    print(idf_weights.shape)
    return text_features, mask, idf_weights

def process(a, tokenizer=None):
    if tokenizer is not None:
        a = tokenizer(a)[0].tolist()
    return set(a)

# def process(a, tokenizer=None):
#     if tokenizer is not None:
       
#         a1 = tokenizer(a)[0].tolist()
        
#         #试试看换成unixcoder模型怎么得到相同的结果
#         tokenizer1=AutoTokenizer.from_pretrained('microsoft/codebert-base')
#         sentence = "This is an example sentence."

#         a2 = tokenizer1.encode(a, add_special_tokens=True, return_tensors="pt").tolist()[0]

#         #a1 = tokenizer1.encode(a)
#         print("--------换成unixcoder")
        
#         print(a2)
#         print(type(a2))
#         print(len(a2))
#     return set(a1)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """

    print("输出get_idf_dict的一些中间信息：")
    idf_count = Counter()
    num_docs = len(arr)

    print(idf_count)
    print(num_docs)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    print(idf_count)

    #这里对idf_dict进行初始化（目前还是空的）
    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))

    #idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})

    print(idf_count.items())

    print(idf_dict)

    return idf_dict


def refs_greedy_cos(ref_embedding, ref_masks, ref_idf, hyp_embedding, hyp_masks, hyp_idf, return_matched_idx):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
    """
    # ref_embedding and hyp_embedding are aleady L2-normalized.
    print("输出一些在计算emscore分数的一些中间输出信息：")
    
    #batch_size是标记reference的个数的
    batch_size = ref_embedding.size(0)
    
    print("batch_size的值是：")
    print(batch_size)

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))

    print(torch.isnan(hyp_embedding).any())
    print(torch.isnan(ref_embedding).any())
    
    print("计算得到的相似度的相关信息：")
    print(sim)
    print(torch.isnan(sim).any())
    print(sim.shape)

    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    
    sim = sim * masks
    print("结合mask之后的sim分数中有nan值吗")
    print(torch.isnan(sim).any())
    
    word_precision, matched_indices = sim.max(dim=2)
    word_recall = sim.max(dim=1)[0]

    print("在每行/列中取最大值的操作结果：")
    #输出的长度是17
    print(word_precision)

    #这个应该是对应选择的最大值的下表
    print(matched_indices)

    #这个值的输出是22
    print(word_recall)

    #这里的ref_idf还没有问题！！！
    print("在进行div_操作之前传进来的ref_idf的值有问题吗")
    print(torch.isnan(ref_idf).any())

    #因为传进来的ref_idf全是0,加和之后也还是0,做div操作之后发生除0错误，tensor的值就变成了nan
    print("没有进行处理的idf值")
    print(hyp_idf)
    print(ref_idf)

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    print("输出ref_idf做sum()之后的值")
    print(ref_idf.sum(dim=1, keepdim=True))

    print("对idf进行一些div操作？")
    print(hyp_idf)
    print(torch.isnan(hyp_idf).any())

    #这个地方出现nan值
    print(ref_idf)
    print(torch.isnan(ref_idf).any())

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)

    F = 2 * P * R / (P + R)
    
    print("return_matched_idx的设置有什么作用？")
    print("如果这个值为真，返回的是matched_indices：")
    print(matched_indices)
    print("如果这个值为假，返回的结果是torch.zeros_like(P)：")
    print(torch.zeros_like(P))

    if return_matched_idx:
        return P, R, F, matched_indices
    else:
        return P, R, F, torch.zeros_like(P)

def vid_greedy_cos(ref_embedding, ref_masks, hyp_embedding, hyp_masks, hyp_idf, return_matched_idx):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
    """
    # ref_embedding and hyp_embedding are aleady L2-normalized.

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision, matched_indices = sim.max(dim=2)
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    P = (word_precision * precision_scale).sum(dim=1)
    R = word_recall.sum(dim=1)/ref_masks.sum(dim=1)
    F = 2 * P * R / (P + R)
    
    if return_matched_idx:
        return P, R, F, matched_indices
    else:
        return P, R, F, torch.zeros_like(P)



def em_cos_score(
    model, refs, hyps, ori_cands, ori_refs, vids, vid_feat_cache, tokenizer, idf_dict, preprocess, verbose=True, batch_size=64, device="cuda:0", return_matched_idx=False
):
    """
    Compute EMScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    refs_preds_local = []
    refs_pred_matched_idxs = []
    refs_preds_global = []

    vid_preds_local = []
    vid_pred_matched_idxs = []
    vid_preds_global = []


    """process text"""
    #这部分的作用应该是选择短的句子放在前面
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    #sentences = dedup_and_sort(hyps)

    print("一些关于encode_text的输入sen_batch相关的信息")
    print(sentences)
    print(type(sentences))

    embs = []
    iter_range = range(0, len(sentences), batch_size)

    print("iter_range的值是：")
    print(iter_range)

    if verbose:
        print("computing text embedding.")
        iter_range = tqdm(iter_range)
    text_local_stats_dict = dict()
    text_global_stats_dict = dict()
    for batch_start in iter_range:
        print(batch_start)
        sen_batch = sentences[batch_start: batch_start + batch_size]
        
        print(sen_batch)
        #实际处理文本内容在这里调用encode_text
        #此处将encode_text函数替换成处理代码任务中candidate的函数
        embs, masks, text_idfs = encode_text(sen_batch, model, tokenizer, idf_dict, device=device)
        embs = embs.cpu()
        masks = masks.cpu()

        #i枚举sentence里的个数，一个reference+一个candidate就是从0到1

        for i, sen in enumerate(sen_batch):
            print(i)
            print(sen)
            sequence_len = masks[i].sum().item()
            
            print("sequence_len的值是：")
            print(sequence_len)

            # For special tokens, use [SOS] and [EOS]
            local_emb = embs[i, 0:sequence_len]
            global_emb = embs[i, sequence_len-1]

            #这里的idf已经全是0了！！！！
            idf = text_idfs[i, 0:sequence_len]

            print("此处获得local embedding向量：")
            print(local_emb)
            print(local_emb.shape)
            print(" 此处获得global embedding向量：")
            print(global_emb)
            print(global_emb.shape)
            print("获得的idf值：")
            print(idf)
            print(idf.shape)

            # For special tokens, don't use any
            # local_emb = embs[i, 1:sequence_len+1]
            # global_emb = embs[i, sequence_len+1]
            # idf = text_idfs[i, 1:sequence_len+1]

            # For special tokens, only use [EOS] 
            # local_emb = embs[i, 1:sequence_len+1]
            # global_emb = embs[i, sequence_len]
            # idf = text_idfs[i, 1:sequence_len+1]

            text_local_stats_dict[sen] = (local_emb, idf)
            text_global_stats_dict[sen] = global_emb

            
    

    """process video"""
    if vids:
        if vid_feat_cache:
            ori_vids = vids
            vid_local_stats_dict = vid_feat_cache
            vid_global_stats_dict = dict()
            for vid in vid_local_stats_dict:
                image_features = vid_local_stats_dict[vid]
                vid_feature = normalize_matrix(torch.mean(image_features, dim=0, keepdim=True)).squeeze()
                vid_global_stats_dict[vid] = vid_feature
        else:
            ori_vids = vids # video paths list
            unique_vids = list(set(vids))
            if verbose:
                print("computing vid embedding.")
            vid_local_stats_dict = dict()
            vid_global_stats_dict = dict()
            for vid_i in tqdm(range(len(unique_vids))):
                video_file = unique_vids[vid_i]

                #此处是对video进行处理的具体实现，调用封装好的encode_video方法
                image_features, vid_feature = encode_video(video_file, preprocess, model, batch_size=512, device=device)
                # vid_name = video_file.split('/')[-1][:-4]
                vid_local_stats_dict[video_file] = image_features.cpu()
                vid_global_stats_dict[video_file] = vid_feature.cpu()


    #对向量进行pad操作
    def pad_local_batch_stats(sen_batch, stats_dict, device):
        #这里传进来的ref_idf就已经全是0了
        # print("pad_local_batch_stats一些中间处理信息：")
        # print(stats_dict)
        stats = [stats_dict[s] for s in sen_batch]
        # print("_-----------")
        # print(stats)
        emb, idf = zip(*stats)
        # print("_-----------")
        # print(idf)
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
        idf_pad = pad_sequence(idf, batch_first=True)
        
        # print("_-----------")
        # print(idf_pad)
        

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    def pad_vid_local_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask
    
    def pad_global_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
        return emb_pad
        
    """ if references are avaliable """
    if refs:
        iter_range = range(0, len(hyps), batch_size)
        if verbose:
            print("computing greedy matching, references as ground truth.")
            iter_range = tqdm(iter_range)

        with torch.no_grad():
            for batch_start in iter_range:
                # print("pad_local/global_batch_stats这个函数处理前：")
                
                batch_hyps = hyps[batch_start: batch_start + batch_size]
                # print(batch_hyps)
                hyp_stats_local = pad_local_batch_stats(batch_hyps, text_local_stats_dict, device)

                # print("pad_local_batch_stats这个函数处理candidate后：")
                # print(hyp_stats_local)
                # print(hyp_stats_local[0])
                # print(hyp_stats_local[0].shape)

                hyp_stats_global = pad_global_batch_stats(batch_hyps, text_global_stats_dict, device)

                # print("pad_global_batch_stats这个函数处理后：")
                # print(hyp_stats_global)

                batch_refs = refs[batch_start: batch_start + batch_size]
                ref_stats_local = pad_local_batch_stats(batch_refs, text_local_stats_dict, device)

                #在这里就已经全都是0了
                # print("pad_local_batch_stats这个函数处理reference后：")
                # print(ref_stats_local)
                # print(ref_stats_local[2])
                # print(ref_stats_local[2].shape)


                ref_stats_global = pad_global_batch_stats(batch_refs, text_global_stats_dict, device)

                #此处开始实际计算emscore的值，调用refs_greedy_cos函数，那在调用这个函数计算得分之前做的处理是干什么的呢？
                P, R, F1, matched_indices = refs_greedy_cos(*ref_stats_local, *hyp_stats_local, return_matched_idx)

                print("输出计算得到的P、R、F1以及matched_idices的值：")
                print(P)
                print(R)
                print(F1)
                print(matched_indices)


                refs_preds_local.append(torch.stack((P, R, F1), dim=-1).cpu())
                refs_pred_matched_idxs.append(matched_indices)

                #这部分是在进行粗粒度匹配分数的计算
                refs_s_cogr = torch.bmm(hyp_stats_global.unsqueeze(1), ref_stats_global.unsqueeze(1).transpose(1,2)).squeeze()


                refs_preds_global.append(refs_s_cogr)
        print("输出计算完之后存结果的列表里面的内容：")
        print(refs_preds_local)
        print(refs_pred_matched_idxs)
        print(refs_preds_global)

    """ if video used as ground truth """
    # if vids:
    #     if verbose:
    #         print("computing greedy matching, video as ground truth.")
    #     iter_range = range(0, len(ori_cands), batch_size)    
    #     with torch.no_grad():
    #         for batch_start in iter_range: 
    #             batch_ori_hyp = ori_cands[batch_start: batch_start + batch_size]
    #             ori_hyp_stats_local = pad_local_batch_stats(batch_ori_hyp, text_local_stats_dict, device)
    #             ori_hyp_stats_global = pad_global_batch_stats(batch_ori_hyp, text_global_stats_dict, device)

    #             batch_ori_vids = ori_vids[batch_start: batch_start + batch_size]
    #             ori_vids_stats_local = pad_vid_local_batch_stats(batch_ori_vids, vid_local_stats_dict, device)
    #             ori_vids_stats_global = pad_global_batch_stats(batch_ori_vids, vid_global_stats_dict, device)

    #             P, R, F1, matched_indices = vid_greedy_cos(*ori_vids_stats_local, *ori_hyp_stats_local, return_matched_idx)
    #             vid_preds_local.append(torch.stack((P, R, F1), dim=-1).cpu())
    #             vid_pred_matched_idxs.append(matched_indices)

    #             vid_s_cogr = torch.bmm(ori_hyp_stats_global.unsqueeze(1), ori_vids_stats_global.unsqueeze(1).transpose(1, 2)).squeeze()
    #             vid_preds_global.append(vid_s_cogr)  


    results = dict()
    """ if references are avaliable """
    if refs:
        refs_preds_local = torch.cat(refs_preds_local, dim=0).cpu()
        if len(refs) != 1:
            refs_preds_global = torch.cat(refs_preds_global, dim=0).cpu()
        else:
            refs_preds_global = refs_preds_global[0].cpu()
        results['refs_result'] = {}
        results['refs_result']['figr'] = refs_preds_local
        results['refs_result']['cogr'] = refs_preds_global
        results['refs_result']['matched_indices'] = torch.cat(refs_pred_matched_idxs)

    """ if video used as ground truth """
    # if vids:
    #     vid_preds_local = torch.cat(vid_preds_local, dim=0).cpu()
    #     if len(vids) != 1:
    #         vid_preds_global = torch.cat(vid_preds_global, dim=0).cpu()
    #     else:
    #         vid_preds_global = vid_preds_global[0].cpu()
    #     results['vid_result'] = {}
    #     results['vid_result']['figr'] = vid_preds_local
    #     results['vid_result']['cogr'] = vid_preds_global
    #     results['vid_result']['matched_indices'] = torch.cat(vid_pred_matched_idxs)


    return results