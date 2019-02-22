import torch
import torch.nn as nn
from torch.autograd import Variable


class PartUpdateEmbedding(nn.Module):

    # update_index come from num_special, 
    # emb_update come from emb_special.  
    # emb_fixed come from emb_word. emb_fixed will be not changed.
    def __init__(self, update_index, emb_update, emb_fixed):
        super(PartUpdateEmbedding, self).__init__()
        self.update_index = update_index
        self.emb_update = emb_update
        self.emb_fixed = emb_fixed
        self.should_update = True
        self.embedding_dim = emb_update.embedding_dim


    # It will update the emb_fixed if should_update is True
    def set_update(self, should_update):
        self.should_update = should_update



    def forward(self, inp):
        assert(inp.dim() == 2)
        # inp.clamp make the value of inp between 0 and 4 since self.update_index - 1 = 4
        # make sure the data into emb_upadate(emb for specials) is special index. 
        r_update = self.emb_update(inp.clamp(0, self.update_index - 1))
        r_fixed = self.emb_fixed(inp)
        
        # compared to lt(self.update_index). if the element of inp < self.update_index, it will be true.
        # you will get the same shape of matrix as inp with boolean value. And then become 0 or 1.
        # unsqueeze(2) will add a one dimension in third index. So the index become: [*][*][0]
        # The shape of r_update is (time_step/sentence_length)*batch*opt.word_vec_size
        # The shape of self.update_index).float().unsqueeze(2) is (time_step/sentence_length)*batch*1, 
        # and then expand to (time_step/sentence_length)*batch*opt.word_vec_size.
        # After expanded, data[*][*][i] equal to data[*][*][j] (0<=i,j<=opt.word_vec_size) and data = inp.data.lt(self.update_index).float().unsqueeze(2).expand_as(r_update)
        mask = Variable(inp.data.lt(self.update_index).float().unsqueeze(
            2).expand_as(r_update), requires_grad=False)

        # mask only contain the value of 0 or 1.
        # After multipy mask, the (result = r_update + r_fixed) means the special token only come from emb_update. 
        # And other words only come from emb_fixed
        r_update = r_update.mul(mask)
        r_fixed = r_fixed.mul(1 - mask)

        if self.should_update:
            return r_update + r_fixed
        else:
            return r_update + Variable(r_fixed.data, requires_grad=False)
