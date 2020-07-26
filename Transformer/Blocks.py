import torch
from Transformer.Modules import MultiHeadAttention, AddNorm, FeedForward

tab = '  |'*1


class EncodingBlock(torch.nn.Module):
    def __init__(self, emb_dim=512, attn_dim=64, n_head=8, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            n_head=n_head
        )
        self.add_norm_1 = AddNorm(dropout=dropout, input_dim=emb_dim)
        self.ff = FeedForward(input_dim=emb_dim, hid_dim=1024)
        self.add_norm_2 = AddNorm(dropout=dropout, input_dim=emb_dim)
    
    def forward(self, enc_input, mask=None):
        add_norm_output = self.add_norm_1(
            enc_input,
            self.attn(enc_input, enc_input, enc_input, mask)
        )
        
        enc_output = self.add_norm_2(
            add_norm_output,
            self.ff(add_norm_output)
        )

        return enc_output


class DecodingBlock(torch.nn.Module):
    def __init__(self, emb_dim=512, attn_dim=64, n_head=8, dropout=0.0):
        super().__init__()
        self.attn_masked = MultiHeadAttention(
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            n_head=n_head
        )
        self.add_norm_1 = AddNorm(dropout=dropout, input_dim=emb_dim)
        self.attn = MultiHeadAttention(
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            n_head=n_head
        )
        self.add_norm_2 = AddNorm(dropout=dropout, input_dim=emb_dim)
        self.ff = FeedForward(input_dim=emb_dim, hid_dim=1024)
        self.add_norm_3 = AddNorm(dropout=dropout, input_dim=emb_dim)
        
    
    def forward(self, enc_output, dec_input, mask1=None, mask2=None):
        add_norm_output = self.add_norm_1(
            dec_input,
            self.attn_masked(dec_input, dec_input, dec_input, mask1)
        )
        print(tab, add_norm_output.size(), enc_output.size())
        add_norm_output = self.add_norm_2(
            add_norm_output,
            self.attn(add_norm_output, enc_output, enc_output, mask2)
        )
        dec_output = self.add_norm_3(
            add_norm_output,
            self.ff(add_norm_output)
        )
        
        return dec_output