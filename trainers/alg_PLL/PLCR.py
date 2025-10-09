from trainers.base_trainer.coop import CoOp_PLL, CoOp
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.base_model import Base_Model
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
from dassl.data import DataManager

import torch.nn.functional as F
from utils.utils_data import corrupt_dataset
from dassl.metrics import compute_accuracy
from utils.utils_data import Augmented_Dataset_Wrapper_PLCR



class plcr_loss(torch.nn.Module): 
    def __init__(self, im_labels):
        self.confidence = im_labels / im_labels.sum(axis=1)[:, None]
        super(plcr_loss,self).__init__()
    
    def forward(self,output1,output2,output3,partialY,index,epoch,max_epoch):
        y_pred_aug0_probas_log = torch.log_softmax(output1, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(output2, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(output3, dim=-1)

        y_pred_aug0_probas = torch.softmax(output1, dim=-1)
        y_pred_aug1_probas = torch.softmax(output2, dim=-1)
        y_pred_aug2_probas = torch.softmax(output3, dim=-1)
        
        consist_loss0 = torch.nn.KLDivLoss(reduction='batchmean')(y_pred_aug0_probas_log, self.confidence[index])
        consist_loss1 = torch.nn.KLDivLoss(reduction='batchmean')(y_pred_aug1_probas_log, self.confidence[index])
        consist_loss2 = torch.nn.KLDivLoss(reduction='batchmean')(y_pred_aug2_probas_log, self.confidence[index])
        
        super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(output1, dim=1)) * (1 - partialY), dim=1))


        # confidence update right after loss
        y_pred_aug0_probas = y_pred_aug0_probas.detach()
        y_pred_aug1_probas = y_pred_aug1_probas.detach()
        y_pred_aug2_probas = y_pred_aug2_probas.detach()

        revisedY0 = partialY.clone()

        revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
        revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(y_pred_aug0_probas.shape[-1], 1).transpose(0, 1)

        self.confidence[index, :] = revisedY0


        lam = min((epoch*2 / max_epoch), 1)

        return lam*(consist_loss0 + consist_loss1 + consist_loss2) + super_loss








@TRAINER_REGISTRY.register()
class CoOp_PLCR(CoOp_PLL):

    def build_data_loader(self):
        dm = DataManager(self.cfg,dataset_wrapper=Augmented_Dataset_Wrapper_PLCR)

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
        return plcr_loss(self.im_labels)
    

    def model_forward_PLL(self,batch):
        input_w1,input_w2,input_s, label, true, index  =  self.parse_batch_train(batch)


        output_w1 = self.model(input_w1)
        output_w2 = self.model(input_w2)
        output_s = self.model(input_s)

        loss = self.loss(output_w1,output_w2,output_s,label,index,self.epoch,self.max_epoch)

        return loss    
    

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input_w1 = batch["img"][0]
            input_w2 = batch["img"][1]
            input_s = batch["img"][2]

            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]

            input_w1,input_w2,input_s,label,true,index = input_w1.to(self.device),input_w2.to(self.device),input_s.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input_w1,input_w2,input_s, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)
    




@TRAINER_REGISTRY.register()
class Palace_PLCR(Palace_PLL):

    def build_data_loader(self):
        dm = DataManager(self.cfg,dataset_wrapper=Augmented_Dataset_Wrapper_PLCR)

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
        return plcr_loss(self.im_labels)
    

    def model_forward_PLL(self,batch):
        input_w1,input_w2,input_s, label, true, index  =  self.parse_batch_train(batch)

        output_w1 = self.model(input_w1)
        output_w2 = self.model(input_w2)
        output_s = self.model(input_s)

        loss = self.loss(output_w1,output_w2,output_s,label,index,self.epoch,self.max_epoch)

        return output_w1, loss    

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input_w1 = batch["img"][0]
            input_w2 = batch["img"][1]
            input_s = batch["img"][2]

            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]

            input_w1,input_w2,input_s,label,true,index = input_w1.to(self.device),input_w2.to(self.device),input_s.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input_w1,input_w2,input_s, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)