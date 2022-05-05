import time
import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
import torchtext
from sklearn.model_selection import train_test_split

from Embedding import ArticleEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--min_item_popularity", default=50, type=int)
parser.add_argument("--min_customer_purchase", default=1, type=int)
parser.add_argument("--LSTM_hidden_size", default=128, type=int)
parser.add_argument("--L1_size", default=128, type=int)
parser.add_argument("--emb_size", default=16, type=int)
parser.add_argument("--Embedding_per_art_feature", default=16, type=int)
parser.add_argument("--max_article_feature_levels", default=500, type=int)

if __name__ == "__main__":
    params = vars(parser.parse_args())
else:
    params = vars(parser.parse_args(args=[]))


class TxnDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        row = self.df.iloc[n, :]
        item = row["item"]  # Raw

        wk = torch.tensor(V_wk.lookup_indices(row["wk"]))
        item = item

        return wk[:-1], item[:-1], item[1:], row["customer_id"]


class TxnModel(pl.LightningModule):
    def __init__(self, E_item, E_wk, V_item, params):
        super().__init__()

        self.E_item = ArticleEmbedding(params)  # Input embedding
        self.E_wk = E_wk
        self.V_item = V_item  # For output layer

        # self.LSTM = torch.nn.LSTM(input_size=self.E_item.embedding_dim + self.E_wk.embedding_dim,
        #                           hidden_size=params["LSTM_hidden_size"], )

        # self.LayerNorm = torch.nn.LayerNorm(normalized_shape=params["LSTM_hidden_size"])
        #
        # self.L = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=params["LSTM_hidden_size"],
        #                     out_features=params["L1_size"]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=params["L1_size"],
        #                     out_features=E_item.num_embeddings),
        # )
        self.Attn = torch.nn.MultiheadAttention(embed_dim=self.E_item.embedding_dim + self.E_wk.embedding_dim,
                                                num_heads=4, batch_first=True)
        self.LayerNorm = torch.nn.LayerNorm(normalized_shape=self.E_item.embedding_dim + self.E_wk.embedding_dim)

        self.L = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.E_item.embedding_dim + self.E_wk.embedding_dim,
                            out_features=params["L1_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=params["L1_size"],
                            out_features=len(V_item)),
        )

        self.CE = torch.nn.CrossEntropyLoss(ignore_index=V_item.get_stoi()["PAD"])
        self.Dropout = torch.nn.Dropout(0)

        self.last_train_loss = 999

    def get_embedding(self, wk, it):
        return torch.cat([self.E_item(it), self.E_wk(wk)], dim=1)

    def forward(self, wk, it: list[list[str]]):
        # item_pack = torch.nn.utils.rnn.pack_sequence([self.get_embedding(w, i)
        #                                               for w, i in zip(wk, it)],
        #                                              enforce_sorted=False)
        #
        # X, (hn, cn) = self.LSTM(item_pack)
        item_pad = torch.nn.utils.rnn.pad_sequence([self.get_embedding(w, i)
                                                    for w, i in zip(wk, it)], batch_first=True)

        padding_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(self.V_item.lookup_indices(i)).to(self.device) for i in it],
            batch_first=True) == 0  # The lookup is just a shortcut to get the padding mask

        X = self.Attn(item_pad, item_pad, item_pad, key_padding_mask=padding_mask, need_weights=False)[0]

        O = self.L(self.Dropout(self.LayerNorm(X)))

        return O

    def get_progress_bar_dict(self):
        d = super().get_progress_bar_dict()
        d["loss"] = f'{self.last_train_loss:.4f}'
        return d

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    @staticmethod
    def acc12_metric(p: torch.tensor, tar: torch.tensor):
        v, idx = torch.topk(p, 12)
        return torch.sum(tar.unsqueeze(1) == idx) / tar.shape[0]

    def validation_step(self, batch, batch_idx):
        wk, it, tar, cid = batch

        tar = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(self.V_item.lookup_indices(t)).to(self.device) for t in tar],
            batch_first=True)

        p = self(wk, it)

        # Ignore PAD and UNK
        p = p[(tar != 0) & (tar != 1)]
        tar = tar[(tar != 0) & (tar != 1)]

        acc = torch.sum((torch.argmax(p, dim=1) == tar).int()) / tar.shape[0]

        self.log("val_loss", self.CE(p, tar), prog_bar=True, batch_size=32)
        self.log("val_acc", acc, prog_bar=True, batch_size=32)
        self.log("val_acc12", self.acc12_metric(p, tar), prog_bar=True, batch_size=32)

    def training_step(self, batch, batch_idx):
        wk, it, tar, cid = batch

        tar = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(self.V_item.lookup_indices(t)).to(self.device) for t in tar],
            batch_first=True)

        p = self(wk, it)
        loss = self.CE(p.reshape((-1, p.shape[-1])), tar.reshape((-1)))

        self.last_train_loss = loss.item()
        self.log("train_loss", loss, logger=True, on_step=True)

        return loss


t = time.time()

df: pd.DataFrame = pd.read_hdf("./cust_week_seq.h5", key="df")
print(f"Data loaded in {time.time() - t:.2f}s")

# Build vocab prior to trimming customers; else the data would be too sparse
V_wk = torchtext.vocab.build_vocab_from_iterator(df["wk"], specials=["PAD"])

V_item = torchtext.vocab.build_vocab_from_iterator(df["item"],
                                                   specials=["PAD", "OOV"],
                                                   min_freq=params["min_item_popularity"])

V_item.set_default_index(V_item.get_stoi()["OOV"])

print(f"Items vocab size: {len(V_item)}")

# len < 1000: avoid OOM
df = df[(df["wk"].apply(len) > params["min_customer_purchase"]) & (df["wk"].apply(len) < 1000)]

print(f"Data length: {len(df)}")

E_wk = torch.nn.Embedding(num_embeddings=len(V_wk), embedding_dim=params["emb_size"])

# Customers that only purchased OOV items. Would cause pack_sequence to fail. Not used for now.
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


M = TxnModel(ArticleEmbedding(params), E_wk, V_item, params)

df_train, df_dev = train_test_split(df, test_size=0.03, random_state=42)

#####
loader_train_full = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train),
    batch_size=params["train_batch_size"], collate_fn=collate
)
loader_dev_full = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_dev),
    batch_size=32, collate_fn=collate
)

#####
loader_train_small = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train.sample(frac=0.05)),
    batch_size=params["train_batch_size"], collate_fn=collate
)

loader_dev_small = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_dev.sample(frac=0.05)),
    batch_size=32, collate_fn=collate
)

#####
loader_toy = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_train[:1000]),
    batch_size=32, collate_fn=collate
)
loader_dev_toy = torch.utils.data.DataLoader(
    dataset=TxnDataset(df_dev[:100]),
    batch_size=32, collate_fn=collate
)

if __name__ == "__main__":
    trainer = pl.Trainer(accelerator="gpu", val_check_interval=0.3)
    trainer.fit(M, train_dataloaders=loader_train_small, val_dataloaders=loader_dev_small, )

    # trainer = pl.Trainer(accelerator="gpu")
    # trainer.fit(M, train_dataloaders=loader_toy, val_dataloaders=loader_toy)
