import time
import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
import torchtext
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--min_item_popularity", default=10, type=int)
parser.add_argument("--min_customer_purchase", default=10, type=int)
parser.add_argument("--LSTM_hidden_size", default=128, type=int)
parser.add_argument("--L1_size", default=64, type=int)
parser.add_argument("--emb_size", default=32, type=int)


params = vars(parser.parse_args())


class TxnDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        row = self.df.iloc[n, :]
        item = torch.tensor(V_item.lookup_indices(row["item"]))

        # Drop all OOV items
        wk = torch.tensor(V_wk.lookup_indices(row["wk"]))  # [item != 0]
        item = item  # [item != 0]

        return wk[:-1], item[:-1], item[1:], row["customer_id"]


class TxnModel(pl.LightningModule):
    def __init__(self, E_item, E_wk, lstm_hidden_size, l1_size):
        super().__init__()

        self.E_item = E_item
        self.E_wk = E_wk

        self.LSTM = torch.nn.LSTM(input_size=E_item.embedding_dim + E_wk.embedding_dim,
                                  hidden_size=lstm_hidden_size, )

        self.LayerNorm = torch.nn.LayerNorm(normalized_shape=lstm_hidden_size)

        self.L = torch.nn.Sequential(
            torch.nn.Linear(in_features=lstm_hidden_size,
                            out_features=l1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=l1_size,
                            out_features=E_item.num_embeddings),
        )

        self.CE = torch.nn.CrossEntropyLoss()
        self.Dropout = torch.nn.Dropout(0.2)

    def get_embedding(self, wk, it):
        return torch.cat([self.E_item(it), self.E_wk(wk)], dim=1)

    def forward(self, wk, it):
        item_pack = torch.nn.utils.rnn.pack_sequence([self.get_embedding(w, i)
                                                      for w, i in zip(wk, it)],
                                                     enforce_sorted=False)
        X, (hn, cn) = self.LSTM(item_pack)

        O = self.L(self.Dropout(self.LayerNorm(X.data)))

        return torch.nn.utils.rnn.PackedSequence(O, X.batch_sizes,
                                                 X.sorted_indices, X.unsorted_indices)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    @staticmethod
    def map12_metric(p: torch.tensor, tar: torch.tensor):
        v, idx = torch.topk(p, 12)
        return (tar == idx.T).sum() / tar.shape[0]

    def validation_step(self, batch, batch_idx):
        wk, it, tar, cid = batch
        tar = torch.cat(tar)

        p = self(wk, it).data

        acc = (torch.argmax(p, dim=1) == tar).cpu().numpy().mean()

        self.log("val_loss", self.CE(p, tar), prog_bar=True, batch_size=32)
        self.log("val_acc", acc, prog_bar=True, batch_size=32)
        self.log("val_map12", self.map12_metric(p, tar), prog_bar=True, batch_size=32)

    def training_step(self, batch, batch_idx):
        wk, it, tar, cid = batch
        tar = torch.cat(tar)

        p = self(wk, it).data
        loss = self.CE(p, tar)

        return loss


t = time.time()

df: pd.DataFrame = pd.read_hdf("./cust_week_seq.h5", key="df")
print(f"Data loaded in {time.time() - t:.2f}s")

# Build vocab prior to trimming customers; else the data would be too sparse
V_wk = torchtext.vocab.build_vocab_from_iterator(df["wk"], specials=["PAD"])

V_item = torchtext.vocab.build_vocab_from_iterator(df["item"],
                                                   specials=["OOV"], min_freq=params["min_item_popularity"])

V_item.set_default_index(V_item.get_stoi()["OOV"])

print(f"Items vocab size: {len(V_item)}")

# Only keep 'influential' customers
df = df[df["wk"].apply(len) > params["min_customer_purchase"]]

print(f"Data length: {len(df)}")

E_item = torch.nn.Embedding(num_embeddings=len(V_item), embedding_dim=params["emb_size"])
E_wk = torch.nn.Embedding(num_embeddings=len(V_wk), embedding_dim=params["emb_size"])

# Customers that only purchased OOV items. Would cause pack_sequence to fail
unk_cid = []


def collate(l):
    item_l = []
    target_l = []
    wk_l = []
    cid_l = []

    global unk_cid
    for wk, item, target, cid in l:
        cid_l.append(cid)
        if wk.shape[0] == 0:
            unk_cid.append(cid)
            continue
        item_l.append(item)
        target_l.append(target)
        wk_l.append(wk)

    return wk_l, item_l, target_l, cid_l


M = TxnModel(E_item, E_wk, params["LSTM_hidden_size"], params["L1_size"])

df_train, df_dev = train_test_split(df, test_size=0.03, random_state=42)

loader_train = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train),
    batch_size=params["train_batch_size"],
    collate_fn=collate
)
loader_dev = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_dev),
    batch_size=32,
    collate_fn=collate
)

loader_toy = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train[:1000]),
    batch_size=32,
    collate_fn=collate
)

loader_dev_toy = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train[:100]),
    batch_size=32,
    collate_fn=collate
)
trainer = pl.Trainer(accelerator="gpu", val_check_interval=0.3)
trainer.fit(M, train_dataloaders=loader_train, val_dataloaders=loader_dev, )

# trainer = pl.Trainer(accelerator="gpu")
# trainer.fit(M, train_dataloaders=loader_toy, val_dataloaders=loader_dev_toy)
