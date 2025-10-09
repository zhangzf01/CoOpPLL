from trainers.base_trainer.coop import CoOp_PLL
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
import torch.nn.functional as F

class cavl_loss(torch.nn.Module):
    def __init__(self):
        super(cavl_loss,self).__init__()


    def forward(self,outputs, partialY):
        cav = outputs * abs(1 - outputs) * partialY
        cav_pred = cav.argmax(axis=1)  
        average_loss =  F.cross_entropy(outputs,cav_pred)      
        return average_loss


@TRAINER_REGISTRY.register()
class CoOp_CAVL(CoOp_PLL):
    def choose_loss_PLL(self):
        return cavl_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return loss    



@TRAINER_REGISTRY.register()
class Palace_CAVL(Palace_PLL):
    def choose_loss_PLL(self):
        return cavl_loss()

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label)
        return output, loss 