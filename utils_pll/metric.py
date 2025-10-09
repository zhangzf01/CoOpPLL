import torch.nn.functional as F
import torch
from dassl.metrics import compute_accuracy


def compute_conf_margin(output,true):
        confidence = F.softmax(output,dim=1)
        seq_index = torch.arange(0, len(confidence)).long()
        true_confidence = confidence[seq_index, true]
        confidence[seq_index, true] = -1
        max_non_true_confidence = torch.max(confidence,dim=1).values
        return torch.mean(true_confidence-max_non_true_confidence)


def compute_MMC(output):
        confidence = F.softmax(output,dim=1)
        max_confidence = torch.max(confidence,dim=1).values
        return torch.mean(max_confidence)

def update_metric(loss,output,label):
        loss_summary={}
        after_epoch_summary={}
        loss_summary['loss'] = loss.item()
        after_epoch_summary['loss'] = loss.item()
        after_epoch_summary['acc']= compute_accuracy(output, label)[0].item()
        # after_epoch_summary["confidence_margin"]=compute_conf_margin(output,label).item()
        # after_epoch_summary['MMC']=compute_MMC(output).item() 
        return loss_summary,after_epoch_summary
