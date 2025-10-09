from trainers.base_trainer.coop import CoOp_PLL
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
import torch.nn.functional as F

class rc_loss(torch.nn.Module): 
    def __init__(self,im_labels):
        super(rc_loss,self).__init__()
            # RC
        tempY = im_labels.sum(dim=1).unsqueeze(1).repeat(1, im_labels.shape[1])
        confidence = im_labels/tempY
        self.confidence = confidence
    

    def forward(self,output, index): 
        logsm_outputs = F.log_softmax(output, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self,model, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = model(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            self.confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            base_value = self.confidence.sum(dim=1).unsqueeze(1).repeat(1, self.confidence.shape[1]) 
            # self.confidence = torch.div(self.confidence, base_value)
            #  for linear probe , to avoid inf loss: self.confidence = torch.div(self.confidence, base_value+1e-7)
            self.confidence = torch.div(self.confidence, base_value+1e-7)



@TRAINER_REGISTRY.register()
class CoOp_RC(CoOp_PLL):
    def choose_loss_PLL(self):
        return rc_loss(self.im_labels)

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,index)
        return loss    
    
    def after_update(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input, label, true, index =  self.parse_batch_train(batch)
            self.loss.confidence_update(self.model, input, label, index)


@TRAINER_REGISTRY.register()
class Palace_RC(Palace_PLL):
    def choose_loss_PLL(self):
        return rc_loss(self.im_labels)

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,index)
        return output, loss    
    
    def after_update(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input, label, true, index =  self.parse_batch_train(batch)
            self.loss.confidence_update(self.model, input, label, index)

@TRAINER_REGISTRY.register()
class LP_RC(CLIP_LP_PLL):
    def choose_loss_PLL(self):
        return rc_loss(self.im_labels)

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,index)
        return loss    
    
    def after_update(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input, label, true, index =  self.parse_batch_train(batch)
            self.loss.confidence_update(self.model, input, label, index)