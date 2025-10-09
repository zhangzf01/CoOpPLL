from trainers.base_trainer.coop import CoOp_PLL
from trainers.base_trainer.palace import Palace_PLL
from trainers.base_trainer.linear_probe import CLIP_LP_PLL
from dassl.engine import TRAINER_REGISTRY
import torch
import torch.nn.functional as F

class lws_loss(torch.nn.Module):
    def __init__(self,im_labels):
        super(lws_loss,self).__init__()
        n, c = im_labels.shape[0], im_labels.shape[1]
        confidence = torch.ones(n, c) / c
        self.confidence = confidence.to(im_labels.device)


    # beta=1, simoid
    # def forward(self,outputs, partialY, index, lw_weight=1, lw_weight0=1):
    #     device = outputs.device

    #     onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    #     onezero[partialY > 0] = 1
    #     counter_onezero = 1 - onezero
    #     onezero = onezero.to(device)
    #     counter_onezero = counter_onezero.to(device)

    #     sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1]).type(outputs.dtype)
    #     sig_loss1 = sig_loss1.to(device)
    #     sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    #     sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
    #         1 + torch.exp(-outputs[outputs > 0]))
    #     l1 = self.confidence[index, :] * onezero * sig_loss1
    #     average_loss1 = torch.sum(l1) / l1.size(0)

    #     sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1]).type(outputs.dtype)
    #     sig_loss2 = sig_loss2.to(device)
    #     sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    #     sig_loss2[outputs < 0] = torch.exp(
    #         outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    #     l2 = self.confidence[index, :] * counter_onezero * sig_loss2
    #     average_loss2 = torch.sum(l2) / l2.size(0)

    #     print(average_loss1.item())
    #     print(average_loss2.item())
    #     average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    #     return average_loss
    
    def forward(self,outputs, partialY, index, lw_weight=1, lw_weight0=1):
        device = outputs.device

        onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
        onezero[partialY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        sm_outputs = F.softmax(outputs, dim=1)

        sig_loss1 = - torch.log(sm_outputs + 1e-8)
        l1 = self.confidence[index, :] * onezero * sig_loss1
        average_loss1 = torch.sum(l1) / l1.size(0)

        sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
        l2 = self.confidence[index, :] * counter_onezero * sig_loss2
        average_loss2 = torch.sum(l2) / l2.size(0)

        average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
        return average_loss



    def confidence_update_lw(self,model, batchX, batchY, batch_index):
        with torch.no_grad():
            device = batchX.device
            batch_outputs = model(batchX)
            sm_outputs = torch.nn.functional.softmax(batch_outputs, dim=1)

            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[batchY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)

            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                self.confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                self.confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2
            self.confidence[batch_index, :] = new_weight



@TRAINER_REGISTRY.register()
class CoOp_LWS(CoOp_PLL):
    def choose_loss_PLL(self):
        return lws_loss(self.im_labels)

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label,index)
        return loss    
    
    def after_update(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input, label, true, index =  self.parse_batch_train(batch)
            self.loss.confidence_update_lw(self.model, input, label, index)


@TRAINER_REGISTRY.register()
class Palace_LWS(Palace_PLL):
    def choose_loss_PLL(self):
        return lws_loss(self.im_labels)

    def model_forward_PLL(self,batch):
        image, label, true,index = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output,label,index)
        return output,loss    
    
    def after_update(self, batch):
        if self.cfg.WSLLEVEL != 0.0:
            input, label, true, index =  self.parse_batch_train(batch)
            self.loss.confidence_update_lw(self.model, input, label, index)
