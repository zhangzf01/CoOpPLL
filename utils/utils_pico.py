import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import copy

class PiCO(nn.Module):

    def __init__(self, num_class, base_encoder, model):
        super().__init__() 
        # basic parameter
        self.num_classes = num_class
        self.low_dim=128
        self.moco_queue=1024
        self.proto_m=0.9
        self.moco_m=0.999

        
        # we allow pretraining for CUB200, or the network will not converge

        self.encoder_q = base_encoder(model, self.num_classes,feat_dim=self.low_dim).cuda()
        # momentum encoder
        self.encoder_k = base_encoder(copy.deepcopy(model), self.num_classes,feat_dim=self.low_dim).cuda()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.moco_queue, self.low_dim).cuda())
        self.register_buffer("queue_pseudo", torch.randn(self.moco_queue).cuda())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).cuda())        
        self.register_buffer("prototypes", torch.zeros(self.num_classes,self.low_dim).cuda())
        self.queue = F.normalize(self.queue, dim=0).cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, im_k=None, partial_Y=None,eval_only=False):
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        predicted_scores = torch.softmax(output.float(), dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        # using partial labels to filter out negative labels

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q.float(), prototypes.t())
        score_prot = torch.softmax(logits_prot.float(), dim=1)

        # update momentum prototypes with pseudo labels
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)):
            self.prototypes[label] = self.prototypes[label]*self.proto_m + (1-self.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # compute key features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder()  # update the momentum encoder
            _, k = self.encoder_k(im_k)


        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels
        torch.set_printoptions(profile="full")
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pseudo_labels_b)

        return output, features, pseudo_labels, score_prot



class BaseEncoder_CLIP(nn.Module):
    def __init__(self,model,num_class, feat_dim=128):
        super().__init__()
        self.model = model
        self.encoder = model.image_encoder

        dim_in  = self.encoder.output_dim
        # mlp as head
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            ).type(self.encoder.conv1.weight.dtype)
        

        self.register_buffer("prototypes", torch.zeros(num_class, feat_dim))
        

    
    def forward(self,x):
        x = x.half()
        feat = self.encoder(x)
        feat_c = self.head(feat)
        logits = self.model(x)
        x = x.float()
        return logits, F.normalize(feat_c, dim=1)
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    always choose one gpu
    """
    return tensor




class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, max_epoch):
        conf_ema_range = '0.95,0.8'
        conf_ema_range = [float(item) for item in conf_ema_range.split(',')]
        start = conf_ema_range[0]
        end = conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / max_epoch * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs.float(), dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
        return None

class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss