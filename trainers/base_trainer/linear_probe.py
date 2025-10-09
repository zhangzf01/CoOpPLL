import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.base_trainer.base_model import Base_Model
from utils.utils_model import load_clip_to_cpu
from utils.utils_data import corrupt_dataset




class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        # no bias because softmax cancels them all 
        self.linear = nn.Linear(input_dim, output_dim,bias=False,dtype=torch.float32)
        self.bn=torch.nn.BatchNorm1d(input_dim, affine=False,dtype=torch.float32)
        nn.init.trunc_normal_(self.linear.weight, mean=0, std=1)

    
    def forward(self, x):
        x = x.float()

        x = self.bn(x)
        x = self.linear(x)
        return x



class CLIP_lin_probe(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

        output_dim = len(classnames)
        input_dim = self.image_encoder.output_dim

        self.head = LinearProbe(input_dim, output_dim)


    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        logits = self.head(image_features)

        return logits


class CLIP_LP(Base_Model):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CLIP_lin_probe(cfg, classnames, clip_model)

        print("Turning off gradients in the image encoder")
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("linear_probe", self.model.head, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        self.build_loss()
        self.clip_model = clip_model
        print(self.loss)


    def build_loss(self):
        raise NotImplementedError
        




class CLIP_LP_PLL(CLIP_LP):

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
        if self.cfg.WSLLEVEL != 0.0:
            self.train_loader_x,self.im_labels=corrupt_dataset(self.train_loader_x,self.cfg.WSLLEVEL)
            self.im_labels=self.im_labels.float().to(self.device)

            
    def model_forward(self, batch): 
        if self.cfg.WSLLEVEL != 0.0:
            return self.model_forward_PLL(batch)
        else: 
            return CLIP_LP.model_forward(self, batch)

    def parse_batch_train(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input = batch["img"]
            label = batch["label"][0]
            true = batch["label"][1]
            index = batch["label"][2]
            input,label,true,index = input.to(self.device),label.to(self.device),true.to(self.device), index.to(self.device)
            return input, label, true, index
        else: 
            return CLIP_LP.parse_batch_train(self, batch)













# ### 还没改!!!

# ##########################################################  variant 1
# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
# }


# # bug: whether to normalize, other per epoch draw

# class CLIP_lin_probe_v1(nn.Module):
#     def __init__(self, cfg, classnames, clip_model,logit_scale):
#         super().__init__()
#         self.image_encoder = clip_model.visual
#         self.dtype = clip_model.dtype

#         output_dim = len(classnames)
#         input_dim = self.image_encoder.output_dim

#         self.head = LinearProbe(input_dim, output_dim)
#         self.logit_scale = logit_scale


#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         logits = self.logit_scale * self.head(image_features)
#         return logits




# @TRAINER_REGISTRY.register()
# class LinearProbe_v1(CLIP_LP_PLL):
#     def build_model(self):
        
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
        
#         if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         print("Building custom CLIP")
#         self.model = CLIP_lin_probe_v1(cfg, classnames, clip_model,clip_model.logit_scale)

#         temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
#         prompts = [temp.format(c.replace("_", " ")) for c in self.dm.dataset.classnames]
#         print(f"Prompts: {prompts}")
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         prompts = prompts.to(self.device)

#         clip_model = clip_model.to(self.device)
#             # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         init_weight_tensor = clip_model.encode_text(prompts)
#         init_weight_tensor = init_weight_tensor / init_weight_tensor.norm(dim=-1, keepdim=True)
        
#         self.model.head.linear.weight =  torch.nn.Parameter(init_weight_tensor)


#         print("Turning off gradients in the image encoder")
#         for name, param in self.model.named_parameters():
#             if "head" not in name:
#                 param.requires_grad_(False)

#         self.model.to(self.device)
#         # NOTE: only give prompt_learner to the optimizer
#         self.optim = build_optimizer(self.model.head, cfg.OPTIM)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
#         self.register_model("head_v1", self.model.head, self.optim, self.sched)

#         self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
#         self.build_loss()

        
        


# ##########################################################  variant 2

# class CLIP_lin_probe_v2(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.image_encoder = clip_model.visual
#         self.dtype = clip_model.dtype

#         output_dim = len(classnames)
#         input_dim = self.image_encoder.output_dim
#         self.head = LinearProbe(input_dim, output_dim)


#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))
#         return self.head(image_features)



# # with no logit scale (v2)

# @TRAINER_REGISTRY.register()
# class LinearProbe_v2(LinearProbe_v1):
#     def build_model(self):
        
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
        
#         if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         print("Building custom CLIP")
#         self.model = CLIP_lin_probe_v2(cfg, classnames, clip_model)

#         temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
#         prompts = [temp.format(c.replace("_", " ")) for c in self.dm.dataset.classnames]
#         print(f"Prompts: {prompts}")
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         prompts = prompts.to(self.device)

#         clip_model = clip_model.to(self.device)
#         init_weight_tensor = clip_model.encode_text(prompts)        
#         self.model.head.linear.weight =  torch.nn.Parameter(init_weight_tensor)


#         print("Turning off gradients in the image encoder")
#         for name, param in self.model.named_parameters():
#             if "head" not in name:
#                 param.requires_grad_(False)

#         self.model.to(self.device)
#         # NOTE: only give prompt_learner to the optimizer
#         self.optim = build_optimizer(self.model.head, cfg.OPTIM)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
#         self.register_model("head_v2", self.model.head, self.optim, self.sched)

#         self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
#         self.build_loss()


