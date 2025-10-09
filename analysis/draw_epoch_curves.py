import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from numpy import transpose
import concat

def match(f_p,metric_arr):
    with open(f_p) as f: 
        arr = f.readlines()
        template1 = re.compile(r'~~~ per epoch ~~~')
        templates = [re.compile(t+r' [-+]?\d+\.\d+ \(([-+]?\d+\.\d+)\)') for t in metric_arr]
        # templates = [re.compile(t+r' \d+\.\d+ \((\d+\.\d+)\)') for t in metric_arr]
        metric_epochs = []
        for ind, text in enumerate(arr):
            if template1.match(text):
                metric_epochs.append([float(t.findall(text)[0]) for t in templates])
    return transpose(metric_epochs).tolist()

def draw_per_epoch(path, metric_list,metric_name,model_name,y1,y2):
    fig,ax = plt.subplots()
    ax.set_ylim(y1,y2)
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric_name)
    for ind,data in enumerate(metric_list):
        ax.plot(range(len(data)),data,label=model_name[ind])
    ax.legend()
    fig.savefig(path, bbox_inches="tight")

    

if __name__ == '__main__':


# different method


    file_path=[
        '/home/zzf/doc/foudation_pll/epoch/ucf101/CoOp_PLL/rn50_16shots/nctx16_cscFalse_ctpend/pll_0.5/seed0/log.txt',
        '/home/zzf/doc/foudation_pll/epoch/ucf101/CoOp_PLL_init/rn50_16shots/nctx16_cscFalse_ctpend/pll_0.5/seed0/log.txt'
    ]

    metric=['test_acc','acc']
    l1 = [match(f,metric)[0] for f in file_path]
    l2 = [match(f,metric)[1] for f in file_path]

    print(l2[0])
    print(l2[1])
    # draw_per_epoch(path1, l1, 'test_accuracy_ucf101_'+pr,  ['coop','coop_init'],0,100)
    # draw_per_epoch(path2, l2, 'training_accuracy_ucf101_'+pr,  ['coop','coop_init'],0,100)

    concat.merge_images(path2, path1, path3)
    