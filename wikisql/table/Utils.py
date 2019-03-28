import os
from path import Path
import torch
import random
import numpy as np
from collections import defaultdict
import inspect

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def sort_for_pack(input_len):
    """ tbl_len is a list of number/tensor
        tbl_len_sorted is descending order of tbl_len. tbl_len is random order.
        tbl_len[idx_sorted[0]] is the biggest number from tbl_len, tbl_len[idx_sorted[0]] is the second biggest.
        so, tbl_len[idx_sorted[i]] = tbl_len_sorted[i]
        so, tbl_len_sorted[idx_map_back[i]] = tbl_len[i]"""
        
    idx_sorted, input_len_sorted = zip(
        *sorted(list(enumerate(input_len)), key=lambda x: x[1], reverse=True))
    idx_sorted, input_len_sorted = list(idx_sorted), list(input_len_sorted)
    idx_map_back = list(map(lambda x: x[0], sorted(
        list(enumerate(idx_sorted)), key=lambda x: x[1])))
    return idx_sorted, input_len_sorted, idx_map_back


def argmax(scores):
    return scores.max(scores.dim() - 1)[1]


def add_pad(b_list, pad_index, return_tensor=True):
    max_len = max((len(b) for b in b_list))
    r_list = []
    for b in b_list:
        r_list.append(b + [pad_index] * (max_len - len(b)))
    if return_tensor:
        return torch.LongTensor(r_list).cuda()
    else:
        return r_list



def tensorToCsv2D(tensor,path=None,token=','):

    def get_variable_name(variable):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is variable]
    tensor = tensor.cpu()
    name = ''.join(get_variable_name(tensor))

    assert(path is not None)

    z = tensor.numpy().tolist()
    if len(np.shape(z)) == 2:
        with open(path,'a') as f:
            f.write(name)
            f.write('\r')
            for i in range(np.shape(z)[0]):
                for j in range(np.shape(z)[1]):
                    f.write(str(z[i][j]))
                    f.write(token)
                f.write('\r')
    elif len(np.shape(z)) == 1:
        with open(path,'a') as f:
            f.write(name)
            f.write('\r')
            for i in range(np.shape(z)[0]):
                f.write(str(z[i]))
                f.write(token)
            f.write('\r')
    else:
        raise "Not support 3D tensor."