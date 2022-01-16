import argparse
import logging
import logging.config

from common.core import Context
from models.qna_classification_model import QnaContentsClassifier

log = logging.getLogger()


def argument_parser():
    parser = argparse.ArgumentParser(
        description="사전학습된 BERT 모델을 이용하여, 고객 QNA 유형을 분류하는 모델을 학습 후 s3에 저장합니다."
    )

    parser.add_argument(
        "-y", "--yml",
        required=True, help="Config YAML path."
    )

    parser.add_argument(
        "--params",
        required=False, help="pipeline parameters.",
        metavar="KEY=VALUE",
        nargs=argparse.ZERO_OR_MORE,
        default=[]
    )
    return parser


def setup_logging(context):
    log_conf = context.conf.get_or_default("logging", {})
    if log_conf:
        logging.config.dictConfig(log_conf)


def main(args=None):
    context = Context(argument_parser().parse_args(args))
    setup_logging(context)
    log.info(f"Model training start. [uid=%s]." % context.uid)

    classifier = QnaContentsClassifier(context)
    train_set, test_set = classifier.load_dataset(context.params.datafile, storage="disk")
    classifier.compile() \
        .train(train_set, mode="fit") \
        .train(test_set, mode="eval") \
        .save(storage="s3", compression=True, delete_save_file=True)

    log.info(f"Model training fin. [uid=%s]." % context.uid)


if __name__ == "__main__":
    main()
