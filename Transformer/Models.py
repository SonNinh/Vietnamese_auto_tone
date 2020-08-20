import torch
from Transformer.Blocks import EncodingBlock, DecodingBlock

tab = '  |'*0

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_sequence_mask(seq):
    len_s = seq.size(1)
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        super().__init__()
        self.emb_table = self.sinusoid_encoding_table(max_seq_len, emb_dim)

    def sinusoid_encoding_table(self, max_seq_len, emb_dim):
        '''
        Visual expalnation: images/Posional_embedding.png
        '''
        emb_table = torch.tensor([[i]*emb_dim for i in range(max_seq_len)], dtype=torch.float)
        
        base = torch.tensor([10000.]*emb_dim)
        exponent = torch.tensor([i//2*2/emb_dim for i in range(emb_dim)])
        emb_table.div(torch.pow(base, exponent))
        emb_table.unsqueeze_(0)

        emb_table[:, :, 0::2] = torch.sin(emb_table[:, :, 0::2])
        emb_table[:, :, 1::2] = torch.cos(emb_table[:, :, 1::2])
        
        return emb_table

    def forward(self, batch_seq):
        assert self.emb_table.size(1) >= batch_seq.size(1), \
            f"Lenght of the sequence ({batch_seq.size(1)}) should be either equal or smaller than max_seq_len ({self.emb_table.size(1)})"
        return batch_seq + self.emb_table[:, :batch_seq.size(1), :batch_seq.size(2)].clone().detach().to(batch_seq.device)


class Encoder(torch.nn.Module):
    def __init__(self,
    n_vocab, emb_dim, pad_idx, max_seq_len,
    dropout, n_block, attn_dim, n_head, feedforward_dim, training=False):
        super().__init__()
        self.word_emb = torch.nn.Embedding(
            n_vocab,
            emb_dim,
            padding_idx=pad_idx
        )
        self.position_emb = PositionEmbedding(max_seq_len, emb_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.encode_stack = torch.nn.ModuleList([
            EncodingBlock(emb_dim, attn_dim, n_head, dropout, feedforward_dim, training=training)
            for i in range(n_block)
        ])
        self.training = training

    def forward(self, src_seq, src_mask):
        emb_output = self.word_emb(src_seq)
        enc_output = self.position_emb(emb_output)
        if self.training:
            enc_output = self.dropout(enc_output)

        for block in self.encode_stack:
            enc_output = block(enc_output, src_mask)

        return enc_output



class Decoder(torch.nn.Module):
    def __init__(self,
    n_vocab, emb_dim, pad_idx, max_seq_len, dropout,
    n_block, attn_dim, n_head, feedforward_dim, training=False):
        super().__init__()
        self.word_emb = torch.nn.Embedding(
            n_vocab,
            emb_dim,
            padding_idx=pad_idx
        )
        self.position_emb = PositionEmbedding(max_seq_len, emb_dim)
        self.decode_stack = torch.nn.ModuleList([
            DecodingBlock(emb_dim, attn_dim, n_head, dropout, feedforward_dim, training=training)
            for _ in range(n_block)
        ])
        self.training = training

    def forward(self, enc_output, dec_input, src_mask, seq_mask):
        emb_output = self.word_emb(dec_input)
        dec_output = self.position_emb(emb_output)

        for block in self.decode_stack:
            dec_output = block(enc_output, dec_output, mask1=seq_mask, mask2=src_mask)

        return dec_output

class Transformer(torch.nn.Module):
    def __init__(self,
    n_vocab_in=100, n_vocab_out=100, emb_dim=512, pad_idx=1,
    max_seq_len=12, dropout=0.0, n_block=3,
    attn_dim=64, n_head=8, training=False):
        super().__init__()
        self.pad_idx = pad_idx
    
        self.encoder = Encoder(
            n_vocab_in, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim, n_head, training=training
        )
        self.decoder = Decoder(
            n_vocab_out, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim, n_head, training=training
        )
        self.proj_out = torch.nn.Linear(emb_dim, n_vocab_out)
    
    def forward(self, src_seq, trg_seq, teacher=1):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        
        enc_output = self.encoder(src_seq, src_mask)
        # print(tab, 'enc_output', enc_output.size())
        # print(enc_output[0])
        if teacher == 1:
            seq_mask = get_pad_mask(trg_seq, self.pad_idx) & get_sequence_mask(trg_seq)
            dec_output = self.decoder(enc_output, trg_seq, src_mask, seq_mask)
            dec_output = self.proj_out(dec_output)
            # dec_output = torch.functional.F.softmax(dec_output, dim=-1)
            # print(tab, 'dec_output', dec_output.size())
        else:
            batch_size = src_seq.size(0)
            dec_input = torch.zeros(batch_size, 1, dtype=torch.long).to(src_seq.device)
            seq_mask = get_pad_mask(dec_input, self.pad_idx) & get_sequence_mask(dec_input)
            seq_mask = seq_mask.to(src_seq.device)
            trg_len = trg_seq.size(1)
            for _ in range(trg_len):
                # print(tab, dec_input.size())
                dec_output = self.decoder(enc_output, dec_input, src_mask, seq_mask)
                # print(tab, 'dec_output', dec_output)
                dec_output = self.proj_out(dec_output)
                _, topi = dec_output[:, -1, :].max(-1)
                topi = topi.view(-1, 1)
                dec_input = torch.cat((dec_input, topi), 1).to(src_seq.device)

        return dec_output

class TransformerInfer(torch.nn.Module):
    def __init__(self,
    n_vocab_in=100, n_vocab_out=100, emb_dim=512, pad_idx=1,
    max_seq_len=12, dropout=0.0, n_block=3,
    attn_dim=64, n_head=8, training=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(
            n_vocab_in, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim, n_head, training=training
        )
        self.decoder = Decoder(
            n_vocab_out, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim, n_head, training=training
        )
        self.proj_out = torch.nn.Linear(emb_dim, n_vocab_out)
    
    def forward(self, src_seq):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        
        enc_output = self.encoder(src_seq, src_mask)
        # print(tab, 'enc_output', enc_output.size())
        # print(enc_output[0])

        batch_size = src_seq.size(0)
        dec_input = torch.tensor([2]).unsqueeze(0).type(torch.long).to(src_seq.device)
        seq_mask = get_pad_mask(dec_input, self.pad_idx) & get_sequence_mask(dec_input)
        seq_mask = seq_mask.to(src_seq.device)
        
        for _ in range(self.max_seq_len):
            # print(tab, dec_input.size())
            dec_output = self.decoder(enc_output, dec_input, src_mask, seq_mask)
            # print(tab, 'dec_output', dec_output)
            dec_output = self.proj_out(dec_output)
            _, topi = dec_output[:, -1, :].max(-1)
            topi = topi.view(-1, 1)
            dec_input = torch.cat((dec_input, topi), 1).to(src_seq.device)
            
        return dec_output


if __name__ == "__main__":
    n_vocab = 100
    emb_dim = 512
    pad_idx = 1
    max_seq_len = 12
    dropout = 0.1
    n_block = 3
    attn_dim = 64

    device = torch.device('cuda:0')
    model = Transformer(dropout=0.0).to(device)

    # x = torch.randint(0, 100, (3, 3)).cuda()
    # y = torch.randint(0, 100, (3, 4)).cuda()

    # pred = model(x, y, teacher=False)
    # print(tab, pred.size())

    print(model.encoder.encode_stack[0].attn.wQ.bias.device)
