
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import MetricMeter
from dassl.metrics import compute_accuracy





class Base_Model(TrainerX):

    def model_forward(self,batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output, label)
        return loss 


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def forward_backward(self, batch):
        self.loss_summary = {}
        loss = self.model_forward(batch)
        self.model_backward_and_update(loss)
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        self.after_update(batch)
        self.loss_summary['loss'] = loss.item()
        return self.loss_summary
    
    def after_update(self,batch):
        pass

    def before_update(self,batch):
        pass

    def model_inference(self, input):
        return self.model(input)


    def before_epoch(self):
        self.metric_per_epoch = MetricMeter()
        self.after_epoch_summary = {}


    # def after_epoch(self):
    #     with torch.no_grad(): 
    #             for batch_idx, batch in enumerate(self.test_loader):
    #                 input, label = self.parse_batch_test(batch)
    #                 output = self.model(input)
    #                 self.after_epoch_summary['test_acc']= compute_accuracy(output, label)[0].item()
    #                 self.metric_per_epoch.update(self.after_epoch_summary)

    #             print('~~~ per epoch ~~~' + f"{self.metric_per_epoch}")
