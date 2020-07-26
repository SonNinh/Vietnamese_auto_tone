import torch
from torch import optim

from Transformer.Models import Transformer


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(-1)[1]
    pred = pred.view(-1)
    gold = gold.contiguous().view(-1)

    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        pred_flat = pred.view(-1, 64001)
        gold_flat = gold.view(-1)
        loss = F.cross_entropy(pred_flat, gold_flat, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_trg(trg):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_on_epoch(model, optimizer, train_loader, batch_size, device):
    model.train()
    train_loss_total = 0
    n_correct_total = 0
    n_word_total =0
    desc = '  - (Training)   '
    n_batch = 0

    tqdm_bar = tqdm(train_loader, mininterval=2, desc=desc)
    for batch in tqdm_bar:
        n_batch += 1
        src_seq = batch.src.to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

        model.zero_grad()
        pre = model(src_seq, trg_seq, teacher=False)
        loss, n_correct, n_word = cal_performance(pre, gold, trg_pad_idx=1)
        train_loss_total += loss.item()
        n_correct_total += n_correct
        n_word_total += n_word
        tqdm_bar.set_description(f'train loss: {loss.item()/batch_size}, acc: {n_correct_total/n_word_total}')
        # quit()
        loss.backward()
        optimizer.step_and_update_lr()
    
    return train_loss_total/(n_batch*batch_size), n_correct_total/n_word_total


def val_on_epoch(model, val_loader, batch_size, device):
    model.eval()
    val_loss_total = 0
    n_correct_total = 0
    n_word_total =0
    desc = '  - (Validation) '
    n_batch = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, mininterval=2, desc=desc):
            n_batch += 1
            src_seq = batch.src.to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

            pre = model(src_seq, trg_seq, teacher=False)
            loss, n_correct, n_word = cal_performance(pre, gold, trg_pad_idx=1)
            val_loss_total += loss.item()
            n_correct_total += n_correct
            n_word_total += n_word
    
    return val_loss_total/(n_batch*batch_size), n_correct_total/n_word_total


def load_data(file_path, batch_size, device):
    data = pickle.load(open(file_path, 'rb'))
    fields = data['fields']
    train = torchtext.data.Dataset(data['train'], fields)
    val = torchtext.data.Dataset(data['valid'], fields)
    # test = torchtext.data.Dataset(data['test'], fields)
    
    train_loader = MyDataLoader(train, batch_size, device=device)
    val_loader = MyDataLoader(val, batch_size, device=device)
    # test_loader = MyDataLoader(test, batch_size, device=device)

    return train_loader, val_loader


def main():
    n_vocab = 100
    emb_dim = 512
    pad_idx = 1
    max_seq_len = 12
    dropout = 0.1
    n_block = 3
    attn_dim = 64

    model = Transformer(
        dropout=0.0
    )

    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-09
    )

    file_path = 'corpora/vietnamese.pkl'
    model_name = '/mnt/data/sonninh/trained_models/seq2seq/vietnamese_epoch_18.chkpt'
    batch_size = 200
    train_loader, val_loader = load_data(file_path, batch_size, device)

    
    num_epoch = 10
    for i in range(start_epoch, num_epoch+start_epoch):
        train_loss, train_acc_word = train_on_epoch(model, optimizer, train_loader, batch_size, device)
        val_loss, val_acc_word = val_on_epoch(model, val_loader, batch_size, device)
        print(f'Epoch: {i},\n\ttrain loss: {train_loss}, val loss: {val_loss},\n\ttrain_acc: {train_acc_word}, val_acc: {val_acc_word}')
        if val_loss <= min_loss:
            min_loss = val_loss
            checkpoint = {'epoch': i, 'val_loss': val_loss, 'train_loss': train_loss, 'model': model.state_dict()}
            torch.save(checkpoint, model_name)
            print(f'\t- [Info] The checkpoint file has been updated at epoch: {i}, with val_loss: {val_loss}')



if __name__ == "__main__":
    main()
