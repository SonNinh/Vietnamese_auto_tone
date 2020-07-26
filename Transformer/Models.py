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
        return batch_seq + self.emb_table[:, :batch_seq.size(1), :batch_seq.size(2)].clone().detach()


class Encoder(torch.nn.Module):
    def __init__(self, n_vocab, emb_dim, pad_idx, max_seq_len, dropout, n_block, attn_dim):
        super().__init__()
        self.word_emb = torch.nn.Embedding(
            n_vocab,
            emb_dim,
            padding_idx=pad_idx
        )
        self.position_emb = PositionEmbedding(max_seq_len, emb_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.encode_stack = [EncodingBlock(emb_dim, attn_dim, 8, dropout)
                                for i in range(n_block)]


    def forward(self, src_seq, src_mask):
        emb_output = self.word_emb(src_seq)
        enc_output = self.position_emb(emb_output)
        enc_output = self.dropout(enc_output)

        
        for block in self.encode_stack:
            enc_output = block(enc_output, src_mask)

        return enc_output



class Decoder(torch.nn.Module):
    def __init__(self, n_vocab, emb_dim, pad_idx, max_seq_len, dropout, n_block, attn_dim):
        super().__init__()
        self.word_emb = torch.nn.Embedding(
            n_vocab,
            emb_dim,
            padding_idx=pad_idx
        )
        self.position_emb = PositionEmbedding(max_seq_len, emb_dim)
        self.decode_stack = [DecodingBlock(emb_dim, attn_dim, 8, dropout)
                                for i in range(n_block)]


    def forward(self, enc_output, dec_input, src_mask, seq_mask):
        emb_output = self.word_emb(dec_input)
        dec_output = self.position_emb(emb_output)

        for block in self.decode_stack:
            dec_output = block(enc_output, dec_output, mask1=seq_mask, mask2=src_mask)

        return dec_output

class Transformer(torch.nn.Module):
    def __init__(self,
    n_vocab = 100, emb_dim = 512, pad_idx = 1,
    max_seq_len = 12, dropout = 0.0, n_block = 3,
    attn_dim = 64):
        super().__init__()
        self.pad_idx = pad_idx
    
        self.encoder = Encoder(
            n_vocab, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim
        )
        self.decoder = Decoder(
            n_vocab, emb_dim, pad_idx, max_seq_len,
            dropout, n_block, attn_dim
        )
        self.proj_out = torch.nn.Linear(emb_dim, n_vocab)
    
    def forward(self, src_seq, trg_seq, teacher=False):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        seq_mask = get_pad_mask(trg_seq, self.pad_idx) & get_sequence_mask(trg_seq)
        
        enc_output = self.encoder(src_seq, src_mask)
        print(tab, 'enc_output', enc_output.size())
        if teacher:
            dec_output = self.decoder(enc_output, trg_seq, src_mask, seq_mask)
            dec_output = torch.functional.F.softmax(self.proj_out(dec_output), dim=-1)
            print(tab, 'dec_output', dec_output.size())
        else:
            batch_size = src_seq.size(0)
            dec_input = torch.zeros(batch_size, 1, dtype=torch.long)
            seq_mask = get_pad_mask(dec_input, self.pad_idx) & get_sequence_mask(dec_input)

            trg_len = trg_seq.size(1)
            for _ in range(trg_len):
                print(tab, dec_input.size())
                dec_output = self.decoder(enc_output, dec_input, src_mask, seq_mask)
                print(tab, 'dec_output', dec_output)
                dec_output = torch.functional.F.softmax(self.proj_out(dec_output), dim=-1)
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

    model = Transformer(dropout=0.0)

    x = torch.randint(0, 100, (3, 3))
    y = torch.randint(0, 100, (3, 4))

    pred = model(x, y, teacher=False)
    print(tab, pred.size())

