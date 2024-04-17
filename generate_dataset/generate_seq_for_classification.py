
import os
import pandas as pd
import random
random.seed(100)
import csv
from argparse import ArgumentParser
import six
import numpy as np
from mindspore.mindrecord import FileWriter
from utils import tokenization


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert task: MPB-CLS")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", type=bool, default=True, help="Whether to run eval on the dev set.")

    args_opt = parser.parse_args()
    return args_opt


class PaddingInputExample():
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures():

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[str(label)] = i

    tokens_a = tokenizer.tokenize(list(example.text_a))

    data_format = example.data_format
    if data_format == 1:
        tokens_a=truncate_seq_pair_1x(tokens_a, max_seq_length - 2)
    else:
        raise "Unknown data format"

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[str(example.label)]
    if ex_index < 20:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):

    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "int32", "shape": [-1]},
        "is_real_example": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0
    random.shuffle(examples)
    for (ex_index, example) in enumerate(examples):
        all_data = []
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_id, dtype=np.int32)
        is_real_example = np.array(feature.is_real_example, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
                "is_real_example": is_real_example}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)


def truncate_seq_pair_1x(tokens_a,  max_length):
    total_length = len(tokens_a)
    if total_length <= max_length:
        return tokens_a
    else:
        tokens_a=tokens_a[:max_length]
        return tokens_a


# This function is not used by this file but is still used by the Colab and people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


class DataProcessor():

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_val_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file):
        print("FILE")
        print(input_file)
        lines=pd.read_csv(input_file)
        return lines


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        raise ValueError("Unsupported string type: %s" % (type(text)))
    if six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        if isinstance(text, unicode):
            return text
        raise ValueError("Unsupported string type: %s" % (type(text)))
    raise ValueError("Not running on Python2 or Python 3?")


class InputExample():

    def __init__(self, guid, text_a,  label=None,data_format=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.data_format=data_format

class MPB_CLS_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_val_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")



    def _create_examples(self, lines, set_type):
        examples = []
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, i)

            text_a = lines['seq'][i]
            data_format=1
            label = int(lines['label'][i])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label,data_format=data_format))
        return examples

def main():
    if not args.do_train and not args.do_val and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    processor = MPB_CLS_Processor()

    label_list = list(set(list(pd.read_csv(os.path.join(args.data_dir, "train.csv"))["label"])))
    label_list.sort()
    print(label_list)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=False)
    if os.path.exists(args.output_dir)==False:
        os.mkdir(args.output_dir)
    if args.do_train:
        print("data_dir:", args.data_dir)
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples), flush=True)

    if args.do_eval:
        val_examples = processor.get_val_examples(args.data_dir)
        val_file = os.path.join(args.output_dir, "val.mindrecord")
        file_based_convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, tokenizer, val_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples), flush=True)

    if args.do_test:
        predict_examples = processor.get_test_examples(args.data_dir)
        predict_file = os.path.join(args.output_dir, "test.mindrecord")
        file_based_convert_examples_to_features(predict_examples, label_list,args.max_seq_length, tokenizer,predict_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(predict_examples), flush=True)


    print("label:")
    print(label_list)

if __name__ == "__main__":
    args = parse_args()
    main()
