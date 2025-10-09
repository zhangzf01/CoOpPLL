from trainers.base_trainer.coop import CoOp_PLL
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
from torch.nn import functional as F


class naive_loss(torch.nn.Module): 
    def __init__(self):
        super(naive_loss,self).__init__()
    
    # 1/abs(Y) * sigma_{ y \in Y} {-log(y)}
    def forward(self,outputs, partialY): 
        sm_outputs = F.softmax(outputs, dim=1)
        im_label = partialY / partialY.sum(dim=1).unsqueeze(1).repeat(1, partialY.shape[1])
        average_loss = - (torch.log(sm_outputs)*im_label).sum(dim=1).mean()
        return average_loss
    


@TRAINER_REGISTRY.register()
class CoOp_NAIVE(CoOp_PLL):
    def choose_loss_PLL(self):
        return naive_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return loss    

@TRAINER_REGISTRY.register()
class LP_NAIVE(CLIP_LP_PLL):
    def choose_loss_PLL(self):
        return naive_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return loss    
    
