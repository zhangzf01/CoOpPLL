from trainers.base_trainer.coop import *
from trainers.zsclip import *


class ZeroshotCLIP(nn.Module):
    def __init__(self,cfg,classnames,clip_model):
        super(ZeroshotCLIP,self).__init__()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

@TRAINER_REGISTRY.register()
class CoOp_PLL_init(CoOp_PLL):
    def before_train(self):
        CoOp_PLL.before_train(self)
        if self.cfg.WSLLEVEL != 0.0:
            clip_model = load_clip_to_cpu(self.cfg).to(self.device)
            zsclip = ZeroshotCLIP(self.cfg,self.dm.dataset.classnames, clip_model).to(self.device)
            for self.batch_idx, batch in enumerate(self.train_loader_x):
                image, label, true,index = self.parse_batch_train(batch)
                self.loss.confidence_update(zsclip, image, label, index)
