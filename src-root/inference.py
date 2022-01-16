import argparse
import logging
import logging.config
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy
from dateutil.tz import tz
from pandas import DataFrame, Categorical

from common.core import Context
from common.sql import get_sqlalchemy_url, select_no_label_qna_contents, update_qna_contents_label
from models.qna_classification_model import QnaContentsClassifier

log = logging.getLogger()


def argument_parser():
    parser = argparse.ArgumentParser(
        description="사전학습된 BERT 모델을 이용하여, 고객 QNA 유형을 분류하는 모델을 로드 후 추론합니다."
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


def get_db_url(context, select):
    vendor = context.conf.get(f"databases.{select}.vendor")
    dbname = context.conf.get(f"databases.{select}.dbname")
    user = context.conf.get(f"databases.{select}.user")
    pwd = quote_plus(context.conf.get(f"databases.{select}.pwd"))
    host = context.conf.get(f"databases.{select}.host")
    port = context.conf.get(f"databases.{select}.port")

    return get_sqlalchemy_url(vendor, user, pwd, host, port, dbname)


def get_label_dict():
    df = DataFrame(
        {
            "label": [
                "출고문의",
                "배송지연문의",
                "배송완료미수령문의",
                "배송지변경문의"
            ]
        }
    )
    df["category"] = Categorical(df["label"])
    df["label"] = df["category"].cat.codes
    return {
        i[0]: i[1]
        for i
        in df.values
    }


def get_dataset(context):
    ro_url = get_db_url(context, "readonly")

    today = datetime.now(tz.gettz("Asia/Seoul"))
    end_date = context.params.end_date if hasattr(context.params, "end_date") else today.strftime("%Y-%m-%d")
    start_date = context.params.start_date if hasattr(context.params, "start_date") \
        else (today - timedelta(30)).strftime("%Y-%m-%d")

    dataset = select_no_label_qna_contents(ro_url, start_date, end_date)

    sep = " "
    for idx, data in enumerate(zip(dataset["id"], dataset["contents"])):
        dataset["contents"][idx] = data[1].replace("\n", sep)

    dataset = DataFrame(dataset).dropna()
    log.info("Dataset size: %s." % len(dataset))
    return dataset


def update_inference_result(context, data):
    main_url = get_db_url(context, "main")

    for label, pk_list in data.items():
        update_qna_contents_label(main_url, label, pk_list)


def main(args=None):
    context = Context(argument_parser().parse_args(args))
    setup_logging(context)
    log.info(f"Model inference start. [uid=%s]." % context.uid)

    labels = get_label_dict()
    df = get_dataset(context)

    classifier = QnaContentsClassifier(context).compile() \
        .load(context.params.model_file, storage="disk")

    loop_cnt = 0
    chunk_size = 500
    while True:
        start_idx = loop_cnt * chunk_size
        end_idx = (loop_cnt + 1) * chunk_size
        chunk = df[start_idx: end_idx]

        if chunk.empty is True:
            break

        ret = classifier.predict(chunk)
        buf = {}
        for i, arr in enumerate(ret):
            position = i + start_idx
            classification = numpy.argmax(arr)
            classification = labels[classification] if classification in labels else "추론불가"

            log.info(
                "%s contents: %s --> %s." % (chunk["id"][position], chunk["contents"][position], classification))

            if classification not in buf:
                buf[classification] = [chunk["id"][position]]
            else:
                buf[classification].append(chunk["id"][position])

        update_inference_result(context, buf)
        loop_cnt += 1

    log.info(f"Model inference fin. [uid=%s]." % context.uid)


if __name__ == "__main__":
    main()
