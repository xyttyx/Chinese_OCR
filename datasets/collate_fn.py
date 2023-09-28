import torch

def collate_fn(batch: list):
    batch_size = len(batch)
    data = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    len_labels = torch.tensor([len(item[1]) for item in batch])
    max_label_length = max(len_labels)
    
    padded_labels = torch.zeros(batch_size, max_label_length, dtype=torch.int32)
    for i in range(batch_size):
        padded_labels[i,:len_labels[i]] = labels[i]

    return data, padded_labels, len_labels
