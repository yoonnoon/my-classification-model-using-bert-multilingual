# my-classification-model-using-ber-multilingual

---------------------------------------
### 실행 방법

---------------------------------------

(1) Server 에서 실행하는 방법 (**CPU/GPU** 가속기 사용 가능)

~~~shell script
$ git clone https://github.com/yoonnoon/my-classification-model-using-bert-multilingual.git

$ cd ./my-classification-model-using-bert-multilingual

# 학습(training) command example
$ pip install pipenv && \
    pipenv install --ignore-pipfile && \
    pipenv run python3 src-root/command.py \
                    --yml resources/config.yml \
                    --pipeline training \
                    --params \
                    datafile="sample_train_dataset.csv"

# 추론(inference) command example
$ pip install pipenv && \
    pipenv install --ignore-pipfile && \
    pipenv run python3 src-root/command.py \
                    --yml resources/config.yml \
                    --pipeline inference \
                    --params \
                    model_file="sample_model.h5"
~~~

(2) Google Colab 에서 실행 (**GPU/CPU** 가속기만 사용 가능, **TPU** 가속기 사용 못함)
1. Google Drive 접속 후 Colaboratory 앱 설치
2. 드라이브에 [qna-classifier.ipynb](./qna-classifier.ipynb) 업로드 & 주피터 노트북 실행
