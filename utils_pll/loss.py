from torch.nn import functional as F
import torch
from .corrupt import corrupt_dataset

def rc_loss(outputs, confidence, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss


def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
        #weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
        confidence = torch.divide(confidence, base_value+1e-5)
    return confidence


class rc_loss(torch.nn.Module): 
    def __init__(self,confidence):
        super(rc_loss,self).__init__()
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
            self.confidence = torch.divide(self.confidence, base_value)
