import torchtext
import torch
import dill as pickle

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models.roberta import RobertaModel
from transformer.DataLoader import MyDataLoader
from torch.nn import Transfor
__author__ = 'Son Ninh'

class BPE():
    bpe_codes = '/home/sonninh/PhoBert-Sentiment-Classification/PhoBERT_base_fairseq/bpe.codes'

def main():
    phoBERT = RobertaModel.from_pretrained('/home/sonninh/PhoBert-Sentiment-Classification/PhoBERT_base_fairseq', checkpoint_file='model.pt')
    phoBERT.eval()  # disable dropout (or leave in train mode to finetune
    
    args = BPE()
    phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

    SRC = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )
    TRG = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )
    fileds = [('src', SRC), ('trg', TRG)]

    datasets = {'val':None, 'train':None, 'test':None}
    for key in datasets.keys():
        f = open(f'corpora/{key}_src_trg.csv', 'r')
        row = f.readline()
        examples =[] 
        n_sample = 20
        while row and n_sample:
            n_sample -= 1
            src_name, trg_name = row.split(',')
            src_name = phoBERT.encode(src_name)
            trg_name = phoBERT.encode(trg_name)
            ex = torchtext.data.Example.fromlist([src_name, trg_name], fileds)
            examples.append(ex)
            row = f.readline()
        f.close()
        datasets[key] = torchtext.data.Dataset(examples, fileds)

    data = {
        'fields': {'src': SRC, 'trg': TRG},
        'train': datasets['train'].examples,
        'valid': datasets['val'].examples,
        'test': datasets['test'].examples
    }
    pickle.dump(data, open('corpora/vietnamese_mini.pkl', 'wb'))

    # train_iter = MyDataLoader(train, 4)

    # for batch in train_iter:
    #     print(batch.src)
    #     print(batch.trg)


if __name__ == "__main__":
    main()

'''
end_day: done data_loader

TO-DO:
- create fake data
- train model
'''