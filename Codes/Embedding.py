#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import torch
import pytorch_lightning as pl
import torchtext


# In[72]:


arts = pd.read_csv("./articles.csv", dtype=str)
arts.set_index("article_id", inplace=True)  # For faster indexing
arts = arts.drop(columns=["detail_desc"])

# In[124]:


class ArticleEmbedding(pl.LightningModule):
    def __init__(self, params: dict=None):
        super().__init__()
        
        self.E = torch.nn.ModuleDict()
        self.V = dict()

        for col in arts.columns:
            l = len(set(arts[col]))
            if l > params["max_article_feature_levels"]:
                continue
            
            self.V[col] = torchtext.vocab.build_vocab_from_iterator(arts[col].apply(lambda x: [x]),
                                                   specials=["PAD"]) # No need for OOV. All features are fixed
            
            self.E[col] = torch.nn.Embedding(num_embeddings=len(self.V[col]),
                                             embedding_dim=params["Embedding_per_art_feature"])
        
        self.embedding_dim = sum(e.embedding_dim for e in self.E.values())
        self.num_embeddings = len(arts)+2
        
    def forward(self, items):
        rows = arts.loc[items]
        
        embs = []
        for col in self.E:
            embs.append(self.E[col](torch.tensor(
                self.V[col].lookup_indices(rows[col].tolist()), device=self.device
            )))
        
        embs = torch.cat(embs, dim=1)
        return embs


