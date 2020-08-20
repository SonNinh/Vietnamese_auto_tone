import torch

a = 0.0
pred = torch.tensor(((1., a, a, a, a), (a, 0., a, a, a), (a, a, a, 0.1, a)))
trg = torch.tensor((0, 1, 3))


loss = torch.nn.functional.cross_entropy(pred, trg)#, reduction='sum')

print(loss)