import os.path as osp

import torch

from dassl.engine import TRAINER_REGISTRY
from trainers.base_trainer.base_model import Base_Model
from dassl.optim import build_optimizer, build_lr_scheduler

from utils.utils_data import corrupt_dataset
from utils.utils_model import CustomCLIP,load_clip_to_cpu




class CoOp(Base_Model):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.build_loss()
        print(self.loss)


    def build_loss(self):
        raise NotImplementedError


class CoOp_PLL(CoOp):


    def choose_loss_PLL(self):
        raise NotImplementedError
    
    def model_forward_PLL(self,batch):
        raise NotImplementedError


    def build_loss(self):
        if self.cfg.WSLLEVEL != 0.0:
            self.loss =  self.choose_loss_PLL()
        else: 
            self.loss = torch.nn.CrossEntropyLoss()

    

    def build_data_loader(self):
        Base_Model.build_data_loader(self)
        # else it's the same as the normal CoOp
        if self.cfg.WSLLEVEL != 0.0:
            self.train_loader_x,self.im_labels=corrupt_dataset(self.train_loader_x,self.cfg.WSLLEVEL)
            self.im_labels=self.im_labels.float().to(self.device)

            
    def model_forward(self, batch): 
        if self.cfg.WSLLEVEL != 0.0:
            return self.model_forward_PLL(batch)
        else: 
            return CoOp.model_forward(self, batch)

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input = batch["img"]
            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]
            input,label,true,index = input.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input, label, true, index
        else: 
            return CoOp.parse_batch_train(self, batch)
