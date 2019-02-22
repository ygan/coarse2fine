import codecs
import json
import random as rnd
from itertools import chain, count
import six

import torch
import torchtext.data
import torchtext.vocab

UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<blank>'
PAD = 1
BOS_WORD = '<s>'
EOS_WORD = '</s>'
SPLIT_WORD = '<|>'
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, SPLIT_WORD]



def get_fields():
    fields = {}
    fields["src"] = torchtext.data.Field(
        pad_token=PAD_WORD, include_lengths=True)
    fields["ent"] = torchtext.data.Field(
        pad_token=PAD_WORD, include_lengths=False)
    fields["agg"] = torchtext.data.Field(
        sequential=False, use_vocab=False, batch_first=True)
    fields["sel"] = torchtext.data.Field(
        sequential=False, use_vocab=False, batch_first=True)
    fields["tbl"] = torchtext.data.Field(
        pad_token=PAD_WORD, include_lengths=True)
    fields["tbl_split"] = torchtext.data.Field(
        use_vocab=False, pad_token=0)
    fields["tbl_mask"] = torchtext.data.Field(
        use_vocab=False, tensor_type=torch.ByteTensor, batch_first=True, pad_token=1)
    fields["lay"] = torchtext.data.Field(
        sequential=False, batch_first=True)
    fields["cond_op"] = torchtext.data.Field(
        include_lengths=True, pad_token=PAD_WORD)
    fields["cond_col"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=0)
    fields["cond_span_l"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=0)
    fields["cond_span_r"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=0)
    fields["cond_col_loss"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=-1)
    fields["cond_span_l_loss"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=-1)
    fields["cond_span_r_loss"] = torchtext.data.Field(
        use_vocab=False, include_lengths=False, pad_token=-1)
    fields["indices"] = torchtext.data.Field(
        use_vocab=False, sequential=False)
    return fields

def _read_annotated_file(opt, js_list, field, filter_ex):
    
    if field in ('sel', 'agg'):
        lines = (line['query'][field] for line in js_list)
    elif field in ('ent',):
        lines = (line['question']['ent'] for line in js_list)
    elif field in ('tbl',):
        def _tbl(line):
            tk_list = [SPLIT_WORD]
            tk_split = '\t' + SPLIT_WORD + '\t'
            tk_list.extend(tk_split.join(
                ['\t'.join(col['words']) for col in line['table']['header']]).strip().split('\t'))
            tk_list.append(SPLIT_WORD)
            return tk_list
        lines = (_tbl(line) for line in js_list)
    elif field in ('tbl_split',):
        def _cum_length_for_split(line):
            len_list = [len(col['words'])
                        for col in line['table']['header']]
            r = [0]
            for i in range(len(len_list)):
                r.append(r[-1] + len_list[i] + 1)
            return r
        lines = (_cum_length_for_split(line) for line in js_list)
    elif field in ('tbl_mask',):
        lines = ([0 for col in line['table']['header']]
                    for line in js_list)
    elif field in ('lay',):
        def _lay(where_list):
            return ' '.join([str(op) for col, op, cond in where_list])
        lines = (_lay(line['query']['conds'])
                    for line in js_list)
    elif field in ('cond_op',):
        lines = ([str(op) for col, op, cond in line['query']['conds']]
                    for line in js_list)
    elif field in ('cond_col',):
        lines = ([col for col, op, cond in line['query']['conds']]
                    for line in js_list)
    elif field in ('cond_span',):
        def _find_span(q_list, where_list):
            r_list = []
            for col, op, cond in where_list:
                tk_list = cond['words']
                # find exact match first
                if len(tk_list) <= len(q_list):
                    match_list = []
                    for st in range(0, len(q_list) - len(tk_list) + 1):
                        if q_list[st:st + len(tk_list)] == tk_list:
                            match_list.append((st, st + len(tk_list) - 1))
                    if len(match_list) > 0:
                        r_list.append(rnd.choice(match_list))
                        continue
                    elif (opt is not None) and opt.span_exact_match:
                        return None
                    else:
                        # do not have exact match, then fuzzy match (w/o considering order)
                        for len_span in range(len(tk_list), len(tk_list) + 2):
                            for st in range(0, len(q_list) - len_span + 1):
                                if set(tk_list) <= set(q_list[st:st + len_span]):
                                    match_list.append(
                                        (st, st + len_span - 1))
                            if len(match_list) > 0:
                                # match spans that are as short as possible
                                break
                        if len(match_list) > 0:
                            r_list.append(rnd.choice(match_list))
                        else:
                            return None
                else:
                    return None
            return r_list

        def _span(q_list, where_list, filter_ex):
            r_list = _find_span(q_list, where_list)
            if (not filter_ex) and (r_list is None):
                r_list = []
                for col, op, cond in where_list:
                    r_list.append((0, 0))
            return r_list
        lines = (_span(line['question']['words'], line['query']
                        ['conds'], filter_ex) for line in js_list)
    elif field in ('cond_mask',):
        lines = ([0 for col, op, cond in line['query']['conds']]
                    for line in js_list)
    else:
        lines = (line[field]['words'] for line in js_list)
    for line in lines:
        yield line


def _construct_examples(lines, side):
    for words in lines:
        example_dict = {side: words}
        yield example_dict


anno_path="/home/yj/Documents/Python/Pytorch/coarse2fine/data_model/wikisql/annotated_ent/train.jsonl"
with codecs.open(anno_path, "r", "utf-8") as corpus_file:
    js_list = [json.loads(line) for line in corpus_file]
    for js in js_list:
        cond_list = list(enumerate(js['query']['conds']))
        # sort by (operation, orginal index)
        # cond_list.sort(key=lambda x: (x[1][1], x[0]))
        cond_list.sort(key=lambda x: x[1][1])
        js['query']['conds'] = [x[1] for x in cond_list] #x[1]


src_data = _read_annotated_file(None, js_list, 'cond_col', True)
src_examples = _construct_examples(src_data, 'cond_col')
lll = list(span_l_examples)


def _map_to_sublist_index(d_list, idx):
        return [([it[idx] for it in d] if (d is not None) else None) for d in d_list]
span_data = list(_read_annotated_file(
    None, js_list, 'cond_span', True))
span_l_examples = _construct_examples(
    _map_to_sublist_index(span_data, 0), 'cond_span_l')
span_r_examples = _construct_examples(
    _map_to_sublist_index(span_data, 1), 'cond_span_r')
span_l_loss_examples = _construct_examples(
    _map_to_sublist_index(span_data, 0), 'cond_span_l_loss')
span_r_loss_examples = _construct_examples(
    _map_to_sublist_index(span_data, 1), 'cond_span_r_loss')

i=0
for ddi in span_data:
    if(ddi is None):
        print("fuck",i)
        break
    i = i + 1


gann=_find_span2(js_list[111]['question']['words'], js_list[111]['query']['conds'])
js_list[0]['question']['words']
js_list[111]['query']['conds']
js_list[111]['query']['conds']
js_list[111]['query']['conds'][1][2]['words']=['243.0']
gann

def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


# examples: one for each src line or (src, tgt) line pair.
examples = [join_dicts(*it) for it in zip(src_examples, span_l_examples, span_r_examples, span_l_loss_examples, span_r_loss_examples)]

len_before_filter = len(examples)
        
#delete some item that contain none value from examples list that 
examples = list(filter
    ( lambda x: 
        all(
            (value is not None for key, value in x.items())
        )
    , examples
    )
)

ex = examples[0]
keys = ex.keys()
fields = get_fields()
fields = [(k, fields[k])
            for k in (list(keys) + ["indices"])]
zop=list([1,3])+['a']


def construct_final(examples):
    for i, ex in enumerate(examples):
        yield torchtext.data.Example.fromlist(
            [ex[k] for k in keys] + [i],
            fields)

def filter_pred(example):
    return True


coe = construct_final(examples)
lll = list(coe)

for (name, field), val in zip(fields, construct_final(examples)):
    print(type(name))
    print(field)
    print(val)
    break
    if field is not None:
        if isinstance(val, six.string_types):
            val = val.rstrip('\n')

for f in fields:
    print(f)
    break



fields = train.fields

merge_list = []
merge_name_list = ('src', 'tbl')
for split in (valid, test, train,):   #split seperately equal to dev, test, train
    for merge_name_it in merge_name_list:
        fields[merge_name_it].build_vocab(
            split, max_size=opt.src_vocab_size, min_freq=0)
        merge_list.append(fields[merge_name_it].vocab)

# build vocabulary only based on the training set
fields["ent"].build_vocab(
    train, max_size=opt.src_vocab_size, min_freq=0)
fields["lay"].build_vocab(
    train, max_size=opt.src_vocab_size, min_freq=0)
fields["cond_op"].build_vocab(
    train, max_size=opt.src_vocab_size, min_freq=0)

# need to know all the words to filter the pretrained word embeddings
merged_vocab = merge_vocabs(merge_list, vocab_size=opt.src_vocab_size)
for merge_name_it in merge_name_list:
    fields[merge_name_it].vocab = merged_vocab

merged = sum([vocab.freqs for vocab in merge_list], Counter())
y=0
for i in range(6):
    y = y + len(merge_list[i])


fields['src'].build_vocab(valid, max_size=opt.src_vocab_size, min_freq=1)
len(fields['src'].vocab.itos)
for i in range(100):
    fields['src'].vocab.itos[len(fields['src'].vocab.itos)-1-i]
fields['src'].vocab.stoi['school/club']
