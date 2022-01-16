import os
import uuid

import keras
import numpy
import tensorflow as tf
import transformers
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.utils.np_utils import to_categorical
from pandas import Categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import TFBertModel, BertTokenizer, BertConfig

from common.components import *
from common.utils import *

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

log.info("Using tf: v.%s, transformers: v.%s, keras: v.%s." % (
    tf.__version__, transformers.__version__, keras.__version__))


class Const:
    BERT_NAME = "bert-base-multilingual-cased"
    BERT_BATCH_SIZE_TABLE = {
        64: 64,
        128: 32,
        256: 16,
        320: 14,
        384: 12,
        512: 6
    }
    MAX_LENGTH_LIST = BERT_BATCH_SIZE_TABLE.keys()
    TEMP_DIR = ".tmp"


class QnaContentsClassifier(BaseMixin):
    __slots__ = (
        "uid", "max_length", "num_classes", "dropout_rate", "epochs",
        "aws_s3", "device", "bert_config", "bert_tokenizer", "bert_model", "model",
        "fit_result", "eval_result"
    )

    def __init__(self, context=None):
        super().__init__()
        os.makedirs(Const.TEMP_DIR, mode=0o744, exist_ok=True)

        self.uid = uuid.uuid4().hex
        self.max_length = 128
        self.num_classes = 5
        self.dropout_rate = 0.3
        self.epochs = 10
        self.aws_s3 = {}

        if context is not None:
            self.uid = context.uid
            self.aws_s3 = context.conf.get_or_default("aws.s3", {})

            if hasattr(context.params, "max_length"):
                self.max_length = int(context.params.max_length)
                prerequisites_check__state_is_true(self.max_length in Const.MAX_LENGTH_LIST)

            if hasattr(context.params, "num_classes"):
                self.num_classes = int(context.params.num_classes)
                prerequisites_check__state_is_true(self.num_classes > 1)

            if hasattr(context.params, "dropout_rate"):
                self.dropout_rate = float(context.params.dropout_rate)
                prerequisites_check__state_is_true(0 < self.dropout_rate < 1)

            if hasattr(context.params, "epochs"):
                self.epochs = int(context.params.epochs)
                prerequisites_check__state_is_true(self.epochs > 0)

        self.device = tf.config.list_logical_devices().pop(0).name
        gpus = tf.config.list_logical_devices("GPU")
        if len(gpus) > 0:  # check available GPU
            self.device = gpus.pop(0).name
        log.info("Device: %s." % self.device)

        self.bert_config = BertConfig.from_pretrained(Const.BERT_NAME)
        self.bert_tokenizer = BertTokenizer.from_pretrained(Const.BERT_NAME, config=self.bert_config)
        self.bert_model = TFBertModel.from_pretrained(Const.BERT_NAME, config=self.bert_config)
        log.info("Pre-trained BERT: %s." % Const.BERT_NAME)

        self.model = None
        self.fit_result = None
        self.eval_result = None

    @property
    def name(self):
        return "QnaContentsClassifier"

    @aspect_logging
    def compile(self, max_length=None, num_classes=None, dropout_rate=None):
        max_length = max_length if max_length is not None else self.max_length
        num_classes = num_classes if num_classes is not None else self.num_classes
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate

        prerequisites_check__state_is_true(max_length in Const.MAX_LENGTH_LIST)
        prerequisites_check__state_is_true(num_classes > 1)
        prerequisites_check__state_is_true(0 < dropout_rate < 1)

        inputs = {
            "input_ids": Input(shape=(max_length,), dtype="int32"),
            "token_type_ids": Input(shape=(max_length,), dtype="int32"),
            "attention_mask": Input(shape=(max_length,), dtype="int32"),
        }
        main_layer = self.bert_model  # load main layer
        if main_layer.trainable is True:
            main_layer.trainable = False  # freeze main layer

        pooled_output = main_layer(inputs).pooler_output
        dropout = Dropout(dropout_rate, name="pooled_output")(pooled_output, training=False)
        outputs = Dense(num_classes, name="classifier", activation="softmax")(dropout)

        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=[
                "categorical_accuracy",
            ]
        )
        self.model = model
        self.model.summary()
        return self

    @aspect_logging
    def train(self, dataset, max_length=None, num_classes=None, epochs=None, *, mode):
        max_length = max_length if max_length is not None else self.max_length
        num_classes = num_classes if num_classes is not None else self.num_classes
        epochs = epochs if epochs is not None else self.epochs

        prerequisites_check__state_is_true(max_length in Const.MAX_LENGTH_LIST)
        prerequisites_check__state_is_true(num_classes > 1)
        prerequisites_check__state_is_true(epochs > 0)
        prerequisites_check__state_is_true(mode in ("fit", "eval"))
        prerequisites_check__state_is_true(self.model is not None)

        batch_size = Const.BERT_BATCH_SIZE_TABLE[max_length]

        x = self._embedding_x(dataset.contents.to_list(), max_length)
        y = self._embedding_y(dataset.label.to_list(), num_classes)

        with tf.device(self.device):
            log.info("Mode: %s" % mode)
            if mode == "fit":
                self.fit_result = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
            else:
                self.eval_result = self.model.evaluate(x=x, y=y, batch_size=batch_size, return_dict=True)
        return self

    @aspect_logging
    def predict(self, dataset, max_length=None):
        max_length = max_length if max_length is not None else self.max_length

        prerequisites_check__state_is_true(max_length in Const.MAX_LENGTH_LIST)
        prerequisites_check__state_is_true(self.model is not None)

        ret = None
        for _ in tqdm(range(1)):
            with tf.device(self.device):
                ret = self.model.predict(self._embedding_x(dataset.contents.to_list(), max_length))
        return ret

    @aspect_logging
    def save(self, storage, compression=False, delete_save_file=False):
        prerequisites_check__state_is_true(self.model is not None)
        prerequisites_check__state_is_true(storage in ("disk", "s3"))

        model_file_path = os.path.join(Const.TEMP_DIR, self.name + "-" + self.uid + ".h5")
        log.info("Save weights: %s." % model_file_path)
        for _ in tqdm(range(1)):
            self.model.save_weights(model_file_path)

        if compression is True:  # .tar.gz
            tar_file_path = os.path.join(Const.TEMP_DIR, self.name + "-" + self.uid + ".tar.gz")
            log.info("Compression: %s." % tar_file_path)
            for _ in tqdm(range(1)):
                make_tarfile(tar_file_path, model_file_path)
                if delete_save_file is True:
                    os.remove(model_file_path)
                model_file_path = tar_file_path

        log.info("Write %s: %s." % (storage, model_file_path))
        if self.aws_s3 and storage == "s3":  # upload s3
            for _ in tqdm(range(1)):
                upload_file_to_s3(model_file_path, os.path.basename(model_file_path), **self.aws_s3)
        elif not self.aws_s3 and storage == "s3":
            raise RuntimeError("Not found s3 configuration.")

        if delete_save_file is True:
            os.remove(model_file_path)
        return self

    @aspect_logging
    def load(self, filepath, storage="disk", delete_load_file=False):
        prerequisites_check__state_is_true(self.model is not None)
        prerequisites_check__state_is_true(storage in ("disk", "s3"))

        filepath = self._download_from_storage(filepath, storage)
        filename, gz = os.path.splitext(filepath)
        filename, tar = os.path.splitext(filename)

        if tar + gz == ".tar.gz":  # is compression file ?
            log.info("Extract: %s." % filepath)
            for _ in tqdm(range(1)):
                extract_tarfile(Const.TEMP_DIR, filepath)
                filepath = os.path.join(Const.TEMP_DIR, os.path.basename(filename) + ".h5")

        if os.path.splitext(filepath)[1] == ".h5" and os.path.exists(filepath):  # only load weights using HDF5 file
            log.info("Load weights: %s." % filepath)
            for _ in tqdm(range(1)):
                with tf.device(self.device):
                    self.model.load_weights(filepath)
        else:
            raise RuntimeError("Not found '.h5' file. input filepath: %s." % filepath)

        if delete_load_file is True:
            os.remove(filepath)
        return self

    @aspect_logging
    def load_dataset(self, filepath, storage="disk", delete_load_file=False, test_size=0.1):
        prerequisites_check__state_is_true(storage in ("disk", "s3"))
        prerequisites_check__state_is_true(os.path.splitext(filepath)[1] == ".csv")
        prerequisites_check__state_is_true(0 < test_size < 1)

        filepath = self._download_from_storage(filepath, storage)
        df = load_data_from_csv(filepath)

        if delete_load_file is True:
            os.remove(filepath)

        if "label" in df:
            df["category"] = Categorical(df["label"])
            df["label"] = df["category"].cat.codes

        return train_test_split(df, test_size=test_size, shuffle=True)

    def _embedding_x(self, contents_list, max_length):
        max_length = max_length if max_length is not None else self.max_length
        prerequisites_check__state_is_true(max_length in Const.MAX_LENGTH_LIST)

        input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
        for contents in contents_list:
            encoding = self.bert_tokenizer.encode_plus(
                contents,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            input_ids_list.append(encoding["input_ids"])
            token_type_ids_list.append(encoding["token_type_ids"])
            attention_mask_list.append(encoding["attention_mask"])

        return {
            "input_ids": numpy.array(input_ids_list, dtype="float32"),
            "token_type_ids": numpy.array(token_type_ids_list, dtype="float32"),
            "attention_mask": numpy.array(attention_mask_list, dtype="float32")
        }

    def _embedding_y(self, label_list, num_classes):
        num_classes = num_classes if num_classes is not None else self.num_classes
        prerequisites_check__state_is_true(num_classes > 1)
        return to_categorical(label_list, num_classes=num_classes)

    def _download_from_storage(self, filepath, storage):
        log.info("Read from %s: %s." % (storage, filepath))
        ret = filepath
        for _ in tqdm(range(1)):
            if self.aws_s3 and storage == "s3":
                download_location = os.path.join(Const.TEMP_DIR, filepath)
                download_file_from_s3(download_location, filepath, **self.aws_s3)
                ret = download_location
            elif not self.aws_s3 and storage == "s3":
                raise RuntimeError("Not found s3 configuration.")
        return ret
