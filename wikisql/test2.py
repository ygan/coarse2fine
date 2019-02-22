from __future__ import division

import os
import argparse

import opts
import torch
import torch.nn as nn
import torchtext

import table
import table.IO

# from tensorboard_logger import Logger
from path import Path

def load_fields(train, valid, checkpoint):
    fields = table.IO.TableDataset.load_fields(
        torch.load(os.path.join(opt.data, 'vocab.pt')))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

    return fields

parser = argparse.ArgumentParser(description='test2.py')

opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
opt.pre_word_vecs = os.path.join(opt.data, 'embedding')

train = torch.load(os.path.join(opt.data, 'train.pt'))
valid = torch.load(os.path.join(opt.data, 'valid.pt'))

if opt.train_from:
    print('Loading checkpoint from %s' % opt.train_from)
    checkpoint = torch.load(
        opt.train_from, map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    # I don't like reassigning attributes of opt: it's not clear
    opt.start_epoch = checkpoint['epoch'] + 1
else:
    checkpoint = None
    model_opt = opt

fields = load_fields(train, valid, checkpoint)

vectors = torchtext.vocab.GloVe(
            name="840B", cache=opt.pre_word_vecs, dim=str(opt.word_vec_size))

fields["src"].vocab.load_vectors(vectors)

fields["src"].build_vocab(train.src, max_size=None, min_freq=0,vectors=vectors)

word_padding_idx = fields["src"].vocab.stoi[table.IO.PAD_WORD]
num_word = len(fields["src"].vocab)
emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

emb_word.weight.data
emb_word.weight.data.copy_(fields["src"].vocab.vectors)


class uuIterator(object):
    def create_batches(self):
        print('fuck')

    def hellp(self):
        print('come')
        self.create_batches()


class uuIterator2(uuIterator):
    def create_batches(self):
        print('fuck222')

z = uuIterator2()
z.hellp()

#############################################
#############################################
#############################################
train_iter = table.IO.OrderedIterator(
        dataset=train, batch_size=opt.batch_size, device=opt.gpuid[0], repeat=False)

breakB = False
for i, batch in enumerate(train_iter):
    break
    for i in range(len(batch.src[1])):
        if(batch.src[1][0]!=batch.src[1][i]):
            breakB = True
            break
    if(breakB):
        break

print(batch.src[0])
dir(batch.tbl[0])
for i in range(10):
    batch.src[0][i][0]
fields['src'].vocab.itos[130]

#########################################
num_word = len(fields['src'].vocab)
word_padding_idx = fields['src'].vocab.stoi[table.IO.PAD_WORD]
emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

num_special = len(table.IO.special_token_list)
# zero vectors in the fixed embedding (emb_word). Actually it is zero in GloVe vector since the token is not a word
emb_word.weight.data[:num_special].zero_()

emb_special = nn.Embedding(
    num_special, opt.word_vec_size, padding_idx=word_padding_idx)
#####emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
torch.cuda.LongTensor
dir(inp)
inp=batch.src[0]
assert(inp.dim() == 2)
inp.clamp(0, num_special - 1)
r_update = emb_special(inp.clamp(0, num_special - 1))
r_fixed = emb_word(inp)
ooo=inp.data.lt(num_special).float().unsqueeze(2)
r_update = r_update.mul(mask)
r_fixed = r_fixed.mul(1 - mask)





if self.should_update:
    return r_update + r_fixed
else:
    return r_update + Variable(r_fixed.data, requires_grad=False)