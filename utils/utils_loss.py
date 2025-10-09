from torch.nn import functional as F
import torch
from .utils_data import corrupt_dataset





class rc_loss(torch.nn.Module): 
    def __init__(self,confidence):
        super(rc_loss,self).__init__()
        self.confidence = confidence
    

    def forward(self,output, index): 
        logsm_outputs = F.log_softmax(output.float(), dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self,model, batchX, batchY, batch_index):
        with torch.no_grad():

            batch_outputs = model(batchX)
            temp_un_conf = F.softmax(batch_outputs.float(), dim=1)
            self.confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            base_value = self.confidence.sum(dim=1).unsqueeze(1).repeat(1, self.confidence.shape[1]) 
            self.confidence = torch.div(self.confidence, base_value)
            for i,j in enumerate(batch_index):
                if j == 0:
                    print(batchY[i])
                    print(temp_un_conf[i])
                    print(self.confidence[j])


class cc_loss(torch.nn.Module): 
    def __init__(self):
        super(cc_loss,self).__init__()
    
    def forward(self,outputs, partialY): 
        sm_outputs = F.softmax(outputs, dim=1)
        final_outputs = sm_outputs * partialY
        average_loss = - torch.log(final_outputs.float().sum(dim=1)).mean()
        return average_loss


