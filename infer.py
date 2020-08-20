import argparse
from os import path
import os
import dill as pickle
import torch
import torch.nn.functional as F
import torchtext
from torch import optim
from torchtext.data import BucketIterator
from tqdm import tqdm
from Transformer.Models import TransformerInfer
from collections import OrderedDict


def patch_src(src):
    src = src.transpose(0, 1)
    return src

def load_data(data_pkl, batch_size):
    data = pickle.load(open(data_pkl, 'rb'))
    fields = data['fields']

    # test = torchtext.data.Dataset(data['test'], fields)
    # test_loader = BucketIterator(test, batch_size, shuffle=True)

    print(fields['src'].vocab.stoi)

    return fields['trg'].vocab.itos, fields['src'].vocab.stoi 


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

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    batch_size = 1
    data_pkl = '/mnt/data/sonninh/vietnamese_tone/pre_processed/mini_vietnamese.pkl'
    itos, stoi = load_data(data_pkl, batch_size)

    model_pth = '/mnt/data/sonninh/trained_models/vietnamese_tone_semi_teacher/best.chkpt'
    checkpoint = torch.load(model_pth)
    opt = checkpoint['opt']

    model = TransformerInfer(
        n_vocab_in=opt.n_vocab_in, n_vocab_out=opt.n_vocab_out,
        emb_dim=opt.emb_dim, pad_idx=opt.pad_idx,
        max_seq_len=opt.max_seq_len, dropout=opt.dropout,
        n_block=opt.n_block, attn_dim=opt.attn_dim, n_head=opt.n_head,
        training=False
    )
    
    model.load_state_dict(
        copyStateDict(checkpoint['model'])
    )
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        while True:
            raw_seq = input('<<<')
            idx_seq = [stoi[c] for c in raw_seq]
            print(idx_seq)
            idx_seq.insert(0, 2)
            idx_seq.append(3)
            src_seq = torch.tensor(idx_seq).type(torch.long).unsqueeze(0).to(device)
            # src_seq = patch_src(idx_seq).to(device)
            pred = model(src_seq)
            pred = pred.max(-1)[1]
            pred = pred.view(-1).tolist()
            pred = ''.join([itos[int(i)] for i in pred])
            print(pred)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-data_pkl', type=str, required=True)

    # parser.add_argument('-epoch', type=int, default=10)
    # parser.add_argument('-batch_size', type=int, default=1)
    # parser.add_argument('-save_dir', type=str, required=True)

    # parser.add_argument('-FT', action='store_true')
    # parser.add_argument('-resume_from', type=str, default='')

    # parser.add_argument('-emb_dim', type=int, default=512)
    # parser.add_argument('-n_vocab_in', type=int, default=100)
    # parser.add_argument('-n_vocab_out', type=int, default=100)
    # parser.add_argument('-max_seq_len', type=int, default=64)
    # parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-n_block', type=int, default=3)
    # parser.add_argument('-attn_dim', type=int, default=64)
    # parser.add_argument('-n_head', type=int, default=8)
    # parser.add_argument('-teacher', action='store_true')
    
    # opt = parser.parse_args()
    # print(opt)
    main()
'''
thiết kế tôi gian<
hà thờ mang phong cách thiết kế pháp
chiến tranh thế giới lần thứ nhất<
trong tại điều khiển trận đầu là thomas
bánh trung là môn ân truyền thống ngày tết
chuyển bay bị hoàn nữa tiếng<eos>
các nhà khoa học đang cô gang điều chẹ vácxin
thay đổi phương pháp học<
phong cách truyền thống<eo
chiến dịch truyền thống
kênh truyền thống nhà nước<
công nghệ thống tin<
hệ thống cấp thoát nước<
tổng tuyên cũ
chiến tranh bùng nổ đỏ mẫu thuận kinh tế
chiếnitranh kết thúc với thắng lợi chờ phế động minh
tổng tỷ số hải lượt trận là 8-2

'''