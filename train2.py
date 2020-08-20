import argparse
import os
from collections import OrderedDict
from os import path

import dill as pickle
import torch
import torch.nn.functional as F
import torchtext
from torch import optim
from torchtext.data import BucketIterator
from tqdm import tqdm

from Transformer.Models import Encoder, get_pad_mask


class TransformerEnc(torch.nn.Module):
    def __init__(self,
    n_vocab_in=100, n_vocab_out=100, emb_dim=512, pad_idx=1,
    max_seq_len=12, dropout=0.0, n_block=3,
    attn_dim=64, n_head=8, feedforward_dim=1024, training=False):
        super().__init__()
        self.pad_idx = pad_idx
    
        self.encoder = Encoder(
            n_vocab_in, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim, n_head, feedforward_dim, training=training
        )
        self.proj_out = torch.nn.Linear(emb_dim, n_vocab_out)
    
    def forward(self, src_seq):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        
        enc_output = self.encoder(src_seq, src_mask)
        output = self.proj_out(enc_output)

        return output
        


def cal_performance(pred, gold, pad_idx, n_vocab_out, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, pad_idx, n_vocab_out, smoothing=smoothing)
    # print(pred[:, :5, :6])
    pred = pred.max(-1)[1]
    # print(pred)
    pred = pred.view(-1)
    gold = gold.contiguous().view(-1)

    non_pad_mask = gold.ne(pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, pad_idx, n_vocab_out, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    # gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        pred_flat = pred.view(-1, n_vocab_out)
        gold_flat = gold.view(-1)
        # print(pred.size())
        # print(gold.size())
        loss = F.cross_entropy(pred_flat, gold_flat, ignore_index=pad_idx, reduction='sum')
    return loss


def patch_src(src):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg):
    trg = trg.transpose(0, 1)
    return trg


def train_on_epoch(model, optimizer, train_loader, opt, device):
    model.train()
    train_loss_total = 0
    n_correct_total = 0
    n_word_total =0
    desc = '  - (Training)   '
    n_batch = 0

    tqdm_bar = tqdm(train_loader, mininterval=2, desc=desc)
    for batch in tqdm_bar:
        n_batch += 1
        src_seq = patch_src(batch.src).to(device)
        trg_seq = patch_trg(batch.trg).to(device)
        
        model.zero_grad()
        pre = model(src_seq)
        loss, n_correct, n_word = cal_performance(pre, trg_seq, opt.pad_idx, opt.n_vocab_out)
        train_loss_total += loss.item()
        n_correct_total += n_correct
        n_word_total += n_word
        tqdm_bar.set_description(f'train loss: {loss.item():.4f}, acc: {n_correct_total/n_word_total:.4f}')
        loss.backward()
        optimizer.step()


    return train_loss_total/(n_batch), n_correct_total/n_word_total


def val_on_epoch(model, val_loader, opt, device):
    model.eval()
    val_loss_total = 0
    n_correct_total = 0
    n_word_total =0
    desc = '  - (Validation) '
    n_batch = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, mininterval=2, desc=desc):
            n_batch += 1
            src_seq = patch_src(batch.src).to(device)
            trg_seq = patch_trg(batch.trg).to(device)
        
            pre = model(src_seq)
            loss, n_correct, n_word = cal_performance(pre, trg_seq, opt.pad_idx, opt.n_vocab_out)
            val_loss_total += loss.item()
            n_correct_total += n_correct
            n_word_total += n_word
    
    return val_loss_total/(n_batch), n_correct_total/n_word_total


def load_data(opt, device):
    data = pickle.load(open(opt.data_pkl, 'rb'))
    fields = data['fields']
    train = torchtext.data.Dataset(data['train'], fields)
    val = torchtext.data.Dataset(data['valid'], fields)
    # test = torchtext.data.Dataset(data['test'], fields)
    
    train_loader = BucketIterator(train, opt.batch_size)#, device=device)
    val_loader = BucketIterator(val, opt.batch_size)#, device=device)
    # test_loader = MyDataLoader(test, batch_size, device=device)
    
    opt.pad_idx = fields['src'].vocab.stoi['<pad>']
    opt.init_idx = fields['src'].vocab.stoi['<init>']
    opt.unk_idx = fields['src'].vocab.stoi['<unk>']
    opt.eos_idx = fields['src'].vocab.stoi['<eos>']
    opt.n_vocab_in = len(fields['src'].vocab)
    opt.n_vocab_out = len(fields['trg'].vocab)

    print(fields['src'].vocab.stoi)
    return train_loader, val_loader

def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train_loader, val_loader = load_data(opt, device)

    model = TransformerEnc(
        n_vocab_in=opt.n_vocab_in, n_vocab_out=opt.n_vocab_out,
        emb_dim=opt.emb_dim, pad_idx=opt.pad_idx,
        max_seq_len=opt.max_seq_len, dropout=opt.dropout,
        n_block=opt.n_block, attn_dim=opt.attn_dim,
        feedforward_dim=opt.feedforward_dim ,
        n_head=opt.n_head, training=True
    )


    if opt.FT:
        print('Fine tune!')
        checkpoint = torch.load(opt.resume_from)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(
            copyStateDict(checkpoint['model'])
        )
    else:
        start_epoch = 0
    
    # model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1]).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.98),
        eps=1e-09
    )

    min_loss = 100000000
    if not path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)

    save_name = path.join(opt.save_dir, 'best.chkpt')

    for i in range(start_epoch, opt.epoch+start_epoch):
        print(f'Epoch: {i}')
        train_loss, train_acc_word = train_on_epoch(
            model, optimizer,
            train_loader, opt, device
        )
        val_loss, val_acc_word = val_on_epoch(
            model,
            val_loader, opt, device
        )
        print(f'Epoch: {i},\n\ttrain loss: {train_loss:.4f}, val loss: {val_loss:.4f},\n\ttrain_acc: {train_acc_word:.4f}, val_acc: {val_acc_word:.4f}')
        if val_loss <= min_loss:
            min_loss = val_loss
            checkpoint = {
                'opt': opt, 'epoch': i, 
                'val_loss': val_loss, 'train_loss': train_loss,
                'model': model.state_dict()
            }
            torch.save(checkpoint, save_name)
            print(f'\t- [Info] The checkpoint file has been updated at epoch: {i}, with val_loss: {val_loss:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', type=str, required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-save_dir', type=str, required=True)

    parser.add_argument('-FT', action='store_true')
    parser.add_argument('-resume_from', type=str, default='')

    parser.add_argument('-emb_dim', type=int, default=512)
    parser.add_argument('-feedforward_dim', type=int, default=1024)
    parser.add_argument('-n_vocab_in', type=int, default=100)
    parser.add_argument('-n_vocab_out', type=int, default=100)
    parser.add_argument('-max_seq_len', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-n_block', type=int, default=3)
    parser.add_argument('-attn_dim', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)

    parser.add_argument('-lr', type=float, default=0.0001)
    
    opt = parser.parse_args()
    print(opt)
    main(opt)
