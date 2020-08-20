import torch
from torch.utils.data import Dataset, DataLoader

# from torchtext.data import Dataset, Iterators, BucketIterator

class MyBatch(object):
    def __init__(self, minibatch, fields, device=None):
        '''
        minibatch: list of torchtext.data.Example
        fileds: dictionary, key(str): name of filed, vallue(torchtext.data.Field): field
        '''
        for name, field in fields.items():
            if field is not None:
                batch = [getattr(x, name) for x in minibatch]
                setattr(self, name, self.pad(batch, pad_id=1, device=device))

    def pad(self, batch, pad_id, device=None):
        '''
        pad_id: int
        '''
        max_len = max(x.size()[0] for x in batch)
        padded = []
        for x in batch:
            diff = max(0, max_len-x.size()[0])
            padded.append(x.tolist()+[pad_id]*diff)
        return torch.tensor(padded)#.to(device)


class MyDataLoader(object):
    def __init__(self, dataset, batch_size, device=None):
        '''
        dataset: torchtext.data.Dataset
        batch_size: int
        device: torch.device
        '''
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device

    def __iter__(self):
        for _, minibatch in enumerate(self.batch()):
            yield MyBatch(minibatch, self.dataset.fields, device=self.device)

    def batch(self, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count

        minibatch, size_so_far = [], 0
        for ex in self.dataset:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == self.batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > self.batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
        if minibatch:
            yield minibatch



