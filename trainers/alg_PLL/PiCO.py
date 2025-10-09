from trainers.base_trainer.coop import CoOp_PLL, CoOp
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.base_model import Base_Model
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
from dassl.data import DataManager
import copy
import torch.nn.functional as F
from utils.utils_data import corrupt_dataset
from dassl.metrics import compute_accuracy
from utils.utils_data import Augmented_Dataset_Wrapper_PiCO
from utils.utils_pico import *

class pico_loss(torch.nn.Module): 
    def __init__(self, im_labels):
        super(pico_loss,self).__init__()

        tempY = im_labels.sum(dim=1).unsqueeze(1).repeat(1, im_labels.shape[1])
        confidence = im_labels/tempY
        confidence = confidence.cuda()
        self.loss_fn = partial_loss(confidence)
        self.loss_cont_fn = SupConLoss()

    
    def forward(self, outputs, index,  features, mask=None, batch_size=-1):
        
        # as the paper suggests!
        loss_weight = 0.5


        loss_cls=self.loss_fn(outputs, index)
        loss_cont=self.loss_cont_fn(features, mask, batch_size)

        loss = loss_cls + loss_weight * loss_cont

        return loss





@TRAINER_REGISTRY.register()
class CoOp_PiCO(CoOp_PLL):
    def build_data_loader(self):
            dm = DataManager(self.cfg,dataset_wrapper=Augmented_Dataset_Wrapper_PiCO)

            self.train_loader_x = dm.train_loader_x
            self.train_loader_u = dm.train_loader_u  # optional, can be None
            self.val_loader = dm.val_loader  # optional, can be None
            self.test_loader = dm.test_loader

            self.num_classes = dm.num_classes
            self.num_source_domains = dm.num_source_domains
            self.lab2cname = dm.lab2cname  # dict {label: classname}
            self.dm = dm

            # else it's the same as the normal CoOp
            if self.cfg.WSLLEVEL != 0.0:
                self.train_loader_x,self.im_labels=corrupt_dataset(self.train_loader_x,self.cfg.WSLLEVEL)
                self.im_labels=self.im_labels.to(self.device)

    def choose_loss_PLL(self):
        self.pico = PiCO(self.num_classes, BaseEncoder_CLIP, self.model)
        return pico_loss(self.im_labels)
    

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input_w = batch["img"][0]
            input_s = batch["img"][1]

            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]

            input_w,input_s,label,true,index = input_w.to(self.device),input_s.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input_w,input_s, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)

    
    def model_forward_PLL(self,batch):
        X_w, X_s, Y, true, index  =  self.parse_batch_train(batch)

        cls_out, features_cont, pseudo_target_cont, score_prot = self.pico(X_w, X_s, Y)

        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)


        if self.epoch>=1:
            self.loss.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # warmup ended
        
        if self.epoch>=1:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels

        else:
            mask = None

        loss = self.loss(cls_out, index,features_cont,mask,batch_size)
        return loss        
    
    def after_epoch(self):
        super().after_epoch()
        self.loss.loss_fn.set_conf_ema_m(self.epoch, self.max_epoch)



@TRAINER_REGISTRY.register()
class Palace_PiCO(Palace_PLL):
    def build_data_loader(self):
            dm = DataManager(self.cfg,dataset_wrapper=Augmented_Dataset_Wrapper_PiCO)

            self.train_loader_x = dm.train_loader_x
            self.train_loader_u = dm.train_loader_u  # optional, can be None
            self.val_loader = dm.val_loader  # optional, can be None
            self.test_loader = dm.test_loader

            self.num_classes = dm.num_classes
            self.num_source_domains = dm.num_source_domains
            self.lab2cname = dm.lab2cname  # dict {label: classname}
            self.dm = dm

            # else it's the same as the normal CoOp
            if self.cfg.WSLLEVEL != 0.0:
                self.train_loader_x,self.im_labels=corrupt_dataset(self.train_loader_x,self.cfg.WSLLEVEL)
                self.im_labels=self.im_labels.to(self.device)

    def choose_loss_PLL(self):
        self.pico = PiCO(self.num_classes, BaseEncoder_CLIP, self.model)
        return pico_loss(self.im_labels)
    

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input_w = batch["img"][0]
            input_s = batch["img"][1]

            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]

            input_w,input_s,label,true,index = input_w.to(self.device),input_s.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input_w,input_s, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)

    
    def model_forward_PLL(self,batch):
        X_w, X_s, Y, true, index  =  self.parse_batch_train(batch)

        cls_out, features_cont, pseudo_target_cont, score_prot = self.pico(X_w, X_s, Y)

        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)


        if self.epoch>=1:
            self.loss.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # warmup ended
        
        if self.epoch>=1:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels

        else:
            mask = None

        loss = self.loss(cls_out, index,features_cont,mask,batch_size)
        return cls_out,loss        
    
    def after_epoch(self):
        super().after_epoch()
        self.loss.loss_fn.set_conf_ema_m(self.epoch, self.max_epoch)


@TRAINER_REGISTRY.register()
class LP_PiCO(CLIP_LP_PLL):

    def build_data_loader(self):
            dm = DataManager(self.cfg,dataset_wrapper=Augmented_Dataset_Wrapper_PiCO)

            self.train_loader_x = dm.train_loader_x
            self.train_loader_u = dm.train_loader_u  # optional, can be None
            self.val_loader = dm.val_loader  # optional, can be None
            self.test_loader = dm.test_loader

            self.num_classes = dm.num_classes
            self.num_source_domains = dm.num_source_domains
            self.lab2cname = dm.lab2cname  # dict {label: classname}
            self.dm = dm

            # else it's the same as the normal CoOp
            if self.cfg.WSLLEVEL != 0.0:
                self.train_loader_x,self.im_labels=corrupt_dataset(self.train_loader_x,self.cfg.WSLLEVEL)
                self.im_labels=self.im_labels.to(self.device)

    def choose_loss_PLL(self):
        self.pico = PiCO(self.num_classes, BaseEncoder_CLIP, self.model)
        return pico_loss(self.im_labels)
    

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input_w = batch["img"][0]
            input_s = batch["img"][1]

            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]

            input_w,input_s,label,true,index = input_w.to(self.device),input_s.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input_w,input_s, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)

    
    def model_forward_PLL(self,batch):
        X_w, X_s, Y, true, index  =  self.parse_batch_train(batch)

        cls_out, features_cont, pseudo_target_cont, score_prot = self.pico(X_w, X_s, Y)

        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)


        if self.epoch>=1:
            self.loss.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # warmup ended
        
        if self.epoch>=1:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels

        else:
            mask = None

        loss = self.loss(cls_out, index,features_cont,mask,batch_size)
        return loss        
    
    def after_epoch(self):
        super().after_epoch()
        self.loss.loss_fn.set_conf_ema_m(self.epoch, self.max_epoch)