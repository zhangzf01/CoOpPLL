from trainers.base_trainer.coop import CoOp_PLL
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
from torch.nn import functional as F


class cc_loss(torch.nn.Module): 
    def __init__(self):
        super(cc_loss,self).__init__()
    
    def forward(self,outputs, partialY): 
        sm_outputs = F.softmax(outputs, dim=1)
        final_outputs = sm_outputs * partialY
        # average_loss = - torch.log(final_outputs.sum(dim=1)).mean()

        #  for linear probe , to avoid inf loss: self.confidence = torch.div(self.confidence, base_value+1e-7)
        average_loss = - torch.log(final_outputs.sum(dim=1)+1e-7).mean()

        return average_loss
    


@TRAINER_REGISTRY.register()
class CoOp_CC(CoOp_PLL):
    def choose_loss_PLL(self):
        return cc_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return loss    
    

@TRAINER_REGISTRY.register()
class Palace_CC(Palace_PLL):
    def choose_loss_PLL(self):
        return cc_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return output,loss    


@TRAINER_REGISTRY.register()
class LP_CC(CLIP_LP_PLL):
    def choose_loss_PLL(self):
        return cc_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return loss    
