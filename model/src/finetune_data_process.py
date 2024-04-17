from src import tokenization
import mindspore as ms

def truncate_seq_pair_1x(tokens_a,  max_length):
    total_length = len(tokens_a)
    if total_length <= max_length:
        return tokens_a
    else:
        tokens_a=tokens_a[:max_length]
        return tokens_a

def generate_predict_seq_1x(data,tokenizer,seq_len,args_opt):
    tokens_a = list(data["seq"])
    id = str(data["id"])
    tokens_a = tokenizer.tokenize(tokens_a)
    tokens_a = truncate_seq_pair_1x(tokens_a, seq_len - 2)
    assert len(tokens_a) <= seq_len - 2
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    assert len(tokens) == len(segment_ids)
    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len
    if "label" in data.keys():
        label_id = data["label"]
    else:
        label_id = -1
    return ms.Tensor([input_ids]),ms.Tensor([input_mask]),ms.Tensor([segment_ids]),ms.Tensor([[label_id]]),id,"".join(tokens_a)

def truncate_seq_pair_2x(tokens_a, tokens_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b

def generate_predict_seq_2x(data,tokenizer,seq_len,args_opt):

    tokens_a = list(data["seq_0"])
    tokens_b = list(data["seq_1"])
    id = str(data["id_0"]) + "\t" + str(data["id_1"])
    tokens_a = tokenizer.tokenize(tokens_a)
    tokens_b = tokenizer.tokenize(tokens_b)
    tokens_a, tokens_b = truncate_seq_pair_2x(tokens_a, tokens_b, seq_len - 3)
    assert len(tokens_a) + len(tokens_b) <= seq_len - 3
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if len(tokens_b)>0:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    assert len(tokens) == len(segment_ids)
    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len

    if "label" in data.keys():
        label_id = data["label"]
    else:
        label_id = -1

    return ms.Tensor([input_ids]),ms.Tensor([input_mask]),ms.Tensor([segment_ids]),ms.Tensor([[label_id]]),id,"".join(tokens_a),"".join(tokens_b)
