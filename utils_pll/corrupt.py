import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def symmetric_noise(labels,noise_rate):
        class_len=len(set(labels))
        indices = np.random.permutation(len(labels))
        for i, idx in enumerate(indices):
            if i < noise_rate * len(labels):
                labels[idx] = np.random.randint(class_len, dtype=np.int32)
        return labels


def symmetric_partialize(train_labels, partial_rate):
    train_labels=torch.tensor(train_labels)
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def corrupt_dataset(train_loader,rate):
    ds = train_loader.dataset.data_source
    labels = [ds[i].label for i in range(len(ds))]
    print('corrupting...')
    im_labels = symmetric_partialize(labels, rate)
    # RC
    tempY = im_labels.sum(dim=1).unsqueeze(1).repeat(1, im_labels.shape[1])
    confidence = im_labels.float()/tempY
    for i in range(len(ds)):
        train_loader.dataset.data_source[i]._label = [im_labels[i],labels[i],i]
    return train_loader,confidence