import torch

tab = '  |'*2

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim=512, attn_dim=64, n_head=8):
        super().__init__()
        self.wQ = torch.nn.Linear(emb_dim, attn_dim*n_head)
        self.wK = torch.nn.Linear(emb_dim, attn_dim*n_head)
        self.wV = torch.nn.Linear(emb_dim, attn_dim*n_head)
        self.attn_dim = attn_dim
        self.word_proj = torch.nn.Linear(attn_dim*n_head, emb_dim)
        self.n_head = n_head

    def forward(self, input_q, input_k, input_v, mask):
        '''
        input_q, input_k and input_v have the same dim of (batch, seq_len, emb_dim)
        '''
        batch_size, query_len, key_len = input_q.size(0), input_q.size(1), input_k.size(1)

        # STEP 1: calculate query, key and value
        # Visual explanation: images/calculate_Q_K_V.png
        # query, key and value have the same dim of (batch, seq_len, attn_dim*n_head)
        query = self.wQ(input_q)
        key = self.wK(input_k)
        value = self.wV(input_v)
        
        # Step 2: Calulate Multi-head-attention
        # Attention visual explanation: images/self_attention.png
        # Multihead visual explanation: images/multihead.png
        # split query, key and value into n_head groups of sub_query, sub_key_sub_value 
        # dim (batch, seq_len, n_head, attn_dim)
        query = query.view(batch_size, query_len, self.n_head, -1)
        key = key.view(batch_size, key_len, self.n_head, -1)
        value = value.view(batch_size, key_len, self.n_head, -1)

        # dim (batch, n_head, seq_len, attn_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # calculate score for attention
        score = torch.matmul(query, key.transpose(-1, -2))
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
        score = torch.nn.functional.softmax(score.div(self.attn_dim**0.5), dim=-1)

        # Concatenate n_head groups
        output = torch.matmul(score, value).transpose(2, 1)
        output = output.contiguous().view(batch_size, query_len, -1)

        output = self.word_proj(output)
        return output


class AddNorm(torch.nn.Module):
    def __init__(self, dropout=0.1, input_dim=512, training=False):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(input_dim, eps=1e-6)
        self.training = training
    
    def forward(self, X, Y):
        if self.training:
            output = self.dropout(Y) + X
        else:
            output = Y + X
        output = self.layer_norm(output)
        return output


class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hid_dim)
        self.linear_2 = torch.nn.Linear(hid_dim, input_dim)

    def forward(self, X):
        output = torch.nn.functional.relu(self.linear_1(X))
        output = self.linear_2(output)
        return output


if __name__ == "__main__":
    layer = MultiHeadAttention().cuda()
    print(layer.wQ.bias.device)