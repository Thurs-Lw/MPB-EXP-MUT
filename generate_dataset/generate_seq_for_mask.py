import os
import csv
import json
from argparse import ArgumentParser
import numpy as np
from mindspore.mindrecord import FileWriter
import pandas as pd

from utils import tokenization

def parse_args():
    parser = ArgumentParser(description="MP-BERT sequence")
    parser.add_argument("--data_dir", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",default=r"F:\bert\paxdb_v5\processed_data\step_10_mask_dataset\fasta_file")
    parser.add_argument("--vocab_file", type=str, default="F:/S500/mindspore/bert/src/generate_mindrecord/vocab_v2.txt",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default=r"F:\bert\paxdb_v5\processed_data\step_10_mask_dataset\fasta_file",
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="mask prob")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", type=bool, default=True, help="Whether to run eval on the dev set.")

    args_opt = parser.parse_args()
    return args_opt


class InputExample():

    def __init__(self,  text_a, mask_lm_positions,mask_lm_ids,mask_lm_weights):
        self.text_a = text_a
        self.mask_lm_positions = mask_lm_positions
        self.mask_lm_ids = mask_lm_ids
        self.mask_lm_weights = mask_lm_weights



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
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 mask_lm_positions,
                    mask_lm_ids,
                    mask_lm_weights,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_lm_positions = mask_lm_positions
        self.mask_lm_ids = mask_lm_ids
        self.mask_lm_weights = mask_lm_weights



class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_val_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



    @classmethod
    def _read_fasta(cls, input_file, quotechar=None):
        seq_info=[]
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    seq_info.append([line.strip(),""])
                else:
                    seq_info[-1][-1]+=line.strip()

        return seq_info

import random
rng = random.Random(42)
class MPB_SEQ_Processor(DataProcessor):
    """Processor for the CLUENER data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_fasta(os.path.join(data_dir, "train.fasta")))

    def get_val_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_fasta(os.path.join(data_dir, "val.fasta")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_fasta(os.path.join(data_dir, "test.fasta")))

    def _create_examples(self, lines):
        """See base class."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = list(line[1])[:args.max_seq_length-2]
            num_to_mask = int(len(text_a) * args.mask_prob)
            mask_lm_positions=rng.sample(range(len(text_a)), num_to_mask)
            mask_lm_positions.sort()
            mask_lm_ids = []
            mask_lm_weights = []

            for i in mask_lm_positions:
                mask_lm_ids.append(text_a[i])
                mask_lm_weights.append(1.0)  # 假设所有被掩码元素的权重都是1.0
                text_a[i] = '[MASK]'

            examples.append(InputExample(text_a=text_a,mask_lm_positions=mask_lm_positions,mask_lm_ids=mask_lm_ids,mask_lm_weights=mask_lm_weights))
        return examples

def truncate_seq_pair_1x(tokens_a,label_a, max_length):
    total_length = len(tokens_a)
    if total_length <= max_length:
        return tokens_a,label_a
    else:
        tokens_a=tokens_a[:max_length]
        label_a=label_a[:max_length]
        return tokens_a,label_a


def convert_single_example(ex_index, example,  max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""


    label_map={"A":0,"C":1,"D":3,"E":4,"G":5,"H":6,"I":7,"L":8,"M":9,"N":10,"O":11,"P":12,"R":13,"S":14,"T":15,"U":16,"V":17,"X":18,"Y":19}


    tokens_a = example.text_a
    masked_lm_positions=example.mask_lm_positions
    masked_lm_ids=example.mask_lm_ids
    masked_lm_weights=example.mask_lm_weights

    tokens_a=tokenizer.tokenize(tokens_a)
    masked_lm_ids=tokenizer.tokenize(masked_lm_ids)

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(tokens_a)
    segment_ids.extend([0]*len(tokens_a))

    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(segment_ids)==len(tokens)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)
    masked_lm_ids = tokenization.convert_tokens_to_ids(args.vocab_file, masked_lm_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)


    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    max_predictions_per_seq = int(args.max_seq_length * args.mask_prob)+1

    masked_lm_positions=[i+1 for i in masked_lm_positions]


    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    if len(masked_lm_positions) != max_predictions_per_seq:
        print(len(masked_lm_positions),max_predictions_per_seq)

    assert len(masked_lm_positions) == max_predictions_per_seq
    assert len(masked_lm_ids) == max_predictions_per_seq
    assert len(masked_lm_weights) == max_predictions_per_seq

    if ex_index < 1:
        print("*** Example ***")
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions]))
        print("masked_lm_ids: %s" % " ".join([str(x) for x in masked_lm_ids]))
        print("masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights]))


    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        mask_lm_positions=masked_lm_positions,
        mask_lm_ids=masked_lm_ids,
        mask_lm_weights=masked_lm_weights,
    )
    return feature


def file_based_convert_examples_to_features(
        examples,  max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a MINDRecord file."""

    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "mask_lm_positions": {"type": "int32", "shape": [-1]},
        "mask_lm_ids": {"type": "int32", "shape": [-1]},
        "mask_lm_weights": {"type": "float32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0
    skip_seq=0

    total_label=[]

    for (ex_index, example) in enumerate(examples):
        all_data = []
        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        mask_lm_positions = np.array(feature.mask_lm_positions, dtype=np.int32)
        mask_lm_ids = np.array(feature.mask_lm_ids, dtype=np.int32)
        mask_lm_weights = np.array(feature.mask_lm_weights, dtype=np.float32)

        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "mask_lm_positions": mask_lm_positions,
                "mask_lm_ids": mask_lm_ids,
                "mask_lm_weights": mask_lm_weights,}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)
    print("skip "+str(skip_seq))

    print(np.sum(total_label)/len(total_label))

def main():
    if not args.do_train and not args.do_val and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processor = MPB_SEQ_Processor()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=False)

    if args.do_train:
        print("data_dir:", args.data_dir)
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(train_examples,
                                                args.max_seq_length, tokenizer,
                                                train_file)

        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples), flush=True)

    if args.do_eval:
        val_examples = processor.get_val_examples(args.data_dir)
        val_file = os.path.join(args.output_dir, "val.mindrecord")
        file_based_convert_examples_to_features(val_examples,
                                                args.max_seq_length, tokenizer,
                                                val_file)
        print("***** Running valing *****")
        print("  Num examples = %d", len(val_examples), flush=True)

    if args.do_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_file = os.path.join(args.output_dir, "test.mindrecord")
        file_based_convert_examples_to_features(test_examples,
                                                args.max_seq_length, tokenizer,
                                                test_file)
        print("***** Running testing *****")
        print("  Num examples = %d", len(test_examples), flush=True)





if __name__ == "__main__":
    args = parse_args()
    main()
