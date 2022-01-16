import argparse

import inference
import training

registry = {
    "training": training,
    "inference": inference,
}


def argument_parser():
    parser = argparse.ArgumentParser(
        description="사전학습된 BERT 모델을 이용하여, fine-tuning 모델을 학습후 저장하거나 로드 후 추론합니다."
    )

    parser.add_argument(
        "-y", "--yml",
        required=True, help="Config YAML path."
    )

    parser.add_argument(
        "-p", "--pipeline",
        required=True, help="Choice what to do.",
        choices=registry.keys()
    )

    parser.add_argument(
        "--params",
        required=False, help="pipeline parameters.",
        metavar="KEY=VALUE",
        nargs=argparse.ZERO_OR_MORE,
        default=[]
    )
    return parser


def main(args=None):
    args = argument_parser().parse_args(args)
    registry[args.job] \
        .main(args=["-y", args.yml, "-p", *args.params])


if __name__ == "__main__":
    main()
