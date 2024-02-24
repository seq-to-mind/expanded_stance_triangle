import re
import os
import json
import pickle
import random
from collections import Counter

import preprocessor as p
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

from emoji import demojize
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
from csv import reader as csv_reader
from global_config import only_binary_polarity


def normalizeToken(token):
    """ only for feature ablation study"""
    if token.startswith("http") or token.startswith("www") or token.startswith("HTTP") or token.startswith("WWW"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        return token


def normalizeTweet(tweet):
    tokens = tweet.split()
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("canot ", "can not ")
        .replace("doesnt ", "does not ")
        .replace("cannot ", "can not ")
        .replace("Can't", "Can not")
        .replace("can't", "can not")
        .replace("ca n't", "can not")
        .replace("ai n't", "ain't")
        .replace("won't ", "will not ")
        .replace("n't ", " not ")
        .replace("n 't ", " not ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return normTweet.split()


def tweet_text_clean(strings, norm_dict, tokenizer):
    clean_data = strings.replace("#SemST", " ").replace("\’", "\'").replace("\‘", "\'").replace("…", "...").replace("\"\"", "\"").replace("&amp;", "&")
    clean_data = normalizeTweet(clean_data, tokenizer)

    for i in range(len(clean_data)):
        if clean_data[i] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i]]
            continue
    clean_data = [v for k, v in enumerate(clean_data) if k == 0 or v != "@USER" or (v == "@USER" and clean_data[k - 1] != "@USER")]
    clean_data = " ".join(clean_data)
    return clean_data


def data_processing_tweet_stance(raw_sample_list):
    label_dict = {"AGAINST": 0, "FAVOR": 1, "NONE": 2, "Neutral": 3}

    tweet_tokenizer = TweetTokenizer()
    """ additional clean up from P-stance author repo """
    with open("data/p_stance/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("data/p_stance/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    """ basic data pre-processing """
    tmp_list = []
    for tmp_i in raw_sample_list:
        tmp_label = tmp_i[0:tmp_i.index(",")]
        tmp_content = tmp_i[tmp_i.index(",") + 1:]

        tmp_target = tmp_content.split("<s>")[0].strip()
        tmp_content = tmp_content.split("<s>")[1]

        tmp_content = re.sub("\s+", " ", tmp_content.replace("\t", " ")).strip()
        if tmp_content[0] == "\"" and tmp_content[-1] == "\"":
            tmp_content = tmp_content[1:-1]
        tmp_list.append([label_dict[tmp_label], tmp_target + " <s> " + tweet_text_clean(tmp_content, normalization_dict, tweet_tokenizer)])
    return tmp_list


def data_reading_tweet_stance(data_path):
    """ read original samples """
    tmp_train_list, tmp_test_list = [], []
    subset_list = os.listdir(data_path)
    for one_type in subset_list:
        one_list = open("/".join([data_path, one_type, "train.csv"]), encoding="utf-8").readlines()
        one_list = [i.strip() for i in one_list if len(i.split()) > 3]
        tmp_train_list.extend(one_list)
        print('[Info] {} instances from {} {} train set'.format(len(one_list), data_path, one_type))

        one_list = open("/".join([data_path, one_type, "test.csv"]), encoding="utf-8").readlines()
        one_list = [i.strip() for i in one_list if len(i.split()) > 3]
        tmp_test_list.extend(one_list)
        print('[Info] {} instances from {} {} test set'.format(len(one_list), data_path, one_type))

    tmp_train_list = data_processing_tweet_stance(tmp_train_list)
    tmp_test_list = data_processing_tweet_stance(tmp_test_list)

    if only_binary_polarity is True:
        tmp_train_list = [i for i in tmp_train_list if i[0] < 2]
        tmp_test_list = [i for i in tmp_test_list if i[0] < 2]

    for i in range(3):
        print(tmp_train_list[i])

    return tmp_train_list, tmp_test_list


def data_processing_P_stance(raw_sample_list):
    label_dict = {"AGAINST": 0, "FAVOR": 1, "NONE": 2, "Neutral": 3}
    tweet_tokenizer = TweetTokenizer()

    """ additional clean up from P-stance author repo """
    with open("data/p_stance/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("data/p_stance/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    """ basic data pre-processing """
    tmp_list = []
    for tmp_i in raw_sample_list:
        tmp_label = tmp_i[tmp_i.rindex(",") + 1:].strip()
        tmp_content = tmp_i[0:tmp_i.rindex(",")]

        tmp_target = tmp_content[tmp_content.rindex(",") + 1:].strip()
        tmp_content = tmp_content[0:tmp_content.rindex(",")]

        tmp_content = re.sub("\s+", " ", tmp_content.replace("\t", " ")).strip()
        if tmp_content[0] == "\"" and tmp_content[-1] == "\"":
            tmp_content = tmp_content[1:-1]
        tmp_list.append([label_dict[tmp_label], tmp_target + " <s> " + tweet_text_clean(tmp_content, normalization_dict, tweet_tokenizer)])

    return tmp_list


def data_reading_P_stance(data_path):
    """ read original samples """
    subset_list = ["bernie", "trump", "biden"]

    tmp_train_list, tmp_dev_list, tmp_test_list = [], [], []
    for one_type in subset_list:
        one_list = open("/".join([data_path, "raw_train_" + one_type + ".csv"]), encoding="utf-8").readlines()
        one_list = [i.strip() for i in one_list if len(i.split()) > 3]
        tmp_train_list.extend(one_list)
        print('[Info] {} instances from {} {} train set'.format(len(one_list), data_path, one_type))

        one_list = open("/".join([data_path, "raw_val_" + one_type + ".csv"]), encoding="utf-8").readlines()
        one_list = [i.strip() for i in one_list if len(i.split()) > 3]
        tmp_dev_list.extend(one_list)
        print('[Info] {} instances from {} {} val set'.format(len(one_list), data_path, one_type))

        one_list = open("/".join([data_path, "raw_test_" + one_type + ".csv"]), encoding="utf-8").readlines()
        one_list = [i.strip() for i in one_list if len(i.split()) > 3]
        tmp_test_list.extend(one_list)
        print('[Info] {} instances from {} {} test set'.format(len(one_list), data_path, one_type))

    tmp_train_list = data_processing_P_stance(tmp_train_list)
    tmp_dev_list = data_processing_P_stance(tmp_dev_list)
    tmp_test_list = data_processing_P_stance(tmp_test_list)

    if only_binary_polarity is True:
        tmp_train_list = [i for i in tmp_train_list if i[0] < 2]
        tmp_dev_list = [i for i in tmp_dev_list if i[0] < 2]
        tmp_test_list = [i for i in tmp_test_list if i[0] < 2]

    for i in range(3):
        print(tmp_train_list[i])

    return tmp_train_list, tmp_dev_list, tmp_test_list


def data_processing_VAST(raw_sample_list):
    tweet_tokenizer = TweetTokenizer()
    """ additional clean up from P-stance author repo """
    with open("data/p_stance/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("data/p_stance/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    """ basic data pre-processing """
    tmp_list = []
    for tmp_i in raw_sample_list:
        tmp_label = tmp_i[0:tmp_i.index(",")]
        tmp_content = tmp_i[tmp_i.index(",") + 1:]

        tmp_target = tmp_content.split("<SEGMENTER>")[0].strip()
        tmp_content = tmp_content.split("<SEGMENTER>")[1]

        tmp_content = re.sub("\s+", " ", tmp_content.replace("\t", " ")).strip()
        tmp_list.append([int(tmp_label), tmp_target + " <s> " + tweet_text_clean(tmp_content, normalization_dict, tweet_tokenizer)])
    return tmp_list


def data_reading_VAST(data_path):
    tmp_train_list, tmp_dev_list, tmp_test_list = [], [], []

    for one_phase in ['train', 'test', "dev"]:
        file_path = f'{data_path}/vast_{one_phase}.csv'
        df = pd.read_csv(file_path)
        # print(f'# VAST {one_phase} examples: {df.shape[0]}')

        topics = df['topic_str'].tolist()
        tweets = df['post'].tolist()
        stances = df['label'].tolist()
        if one_phase == 'test':
            few_shot = df['seen?'].tolist()
            qte = df['Qte'].tolist()
            sarc = df['Sarc'].tolist()
            imp = df['Imp'].tolist()
            mls = df['mlS'].tolist()
            mlt = df['mlT'].tolist()
        else:
            few_shot = np.zeros(df.shape[0])
            qte = np.zeros(df.shape[0])
            sarc = np.zeros(df.shape[0])
            imp = np.zeros(df.shape[0])
            mls = np.zeros(df.shape[0])
            mlt = np.zeros(df.shape[0])

        if one_phase == "train":
            tmp_train_list = [str(stances[i]) + "," + topics[i] + "<SEGMENTER>" + tweets[i] for i in range(len(topics))]
        elif one_phase == "dev":
            tmp_dev_list = [str(stances[i]) + "," + topics[i] + "<SEGMENTER>" + tweets[i] for i in range(len(topics))]
        elif one_phase == "test":
            tmp_test_list = [str(stances[i]) + "," + topics[i] + "<SEGMENTER>" + tweets[i] for i in range(len(topics))]

    tmp_train_list = data_processing_VAST(tmp_train_list)
    tmp_dev_list = data_processing_VAST(tmp_dev_list)
    tmp_test_list = data_processing_VAST(tmp_test_list)

    if only_binary_polarity is True:
        tmp_train_list = [i for i in tmp_train_list if i[0] < 2]
        tmp_dev_list = [i for i in tmp_dev_list if i[0] < 2]
        tmp_test_list = [i for i in tmp_test_list if i[0] < 2]

    print('[Info] {} instances from {} train set'.format(len(tmp_train_list), data_path))
    print('[Info] {} instances from {} dev set'.format(len(tmp_dev_list), data_path))
    print('[Info] {} instances from {} test set'.format(len(tmp_test_list), data_path))

    for i in range(3):
        print(tmp_train_list[i])
    for i in range(3):
        print(tmp_test_list[i])

    return tmp_train_list, tmp_dev_list, tmp_test_list


def data_processing_tweet_Covid(raw_sample_list):
    tweet_tokenizer = TweetTokenizer()
    """ additional clean up from P-stance author repo """
    with open("data/p_stance/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("data/p_stance/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    """ basic data pre-processing """
    tmp_list = []
    for tmp_i in raw_sample_list:
        tmp_label = tmp_i[0:tmp_i.index(",")]
        tmp_content = tmp_i[tmp_i.index(",") + 1:]

        tmp_target = tmp_content.split("<SEGMENTER>")[0].strip()
        tmp_content = tmp_content.split("<SEGMENTER>")[1]

        tmp_content = re.sub("\s+", " ", tmp_content.replace("\t", " ")).strip()
        tmp_list.append([int(tmp_label), tmp_target + " <s> " + tweet_text_clean(tmp_content, normalization_dict, tweet_tokenizer)])
    return tmp_list


def data_reading_tweet_Covid(data_path):
    """ read original samples """
    label_dict = {"AGAINST": 0, "FAVOR": 1, "NONE": 2, "Neutral": 3}

    subset_list = ["face_masks", "fauci", "school_closures", "stay_at_home_orders"]

    tmp_train_list, tmp_dev_list, tmp_test_list = [], [], []
    for one_type in subset_list:
        one_list = csv_reader(open("/".join([data_path, one_type + "_train.csv"]), encoding="utf-8"))
        one_list = [i for i in one_list][1:]
        for one_item in one_list:
            tmp_train_list.append(str(label_dict[one_item[2]]) + "," + one_item[1].replace("_", " ") + "<SEGMENTER>" + one_item[0])
        print('[Info] {} instances from {} {} train set'.format(len(one_list), data_path, one_type))

        one_list = csv_reader(open("/".join([data_path, one_type + "_val.csv"]), encoding="utf-8"))
        one_list = [i for i in one_list][1:]
        for one_item in one_list:
            tmp_dev_list.append(str(label_dict[one_item[2]]) + "," + one_item[1].replace("_", " ") + "<SEGMENTER>" + one_item[0])
        print('[Info] {} instances from {} {} test set'.format(len(one_list), data_path, one_type))

        one_list = csv_reader(open("/".join([data_path, one_type + "_test.csv"]), encoding="utf-8"))
        one_list = [i for i in one_list][1:]
        for one_item in one_list:
            tmp_test_list.append(str(label_dict[one_item[2]]) + "," + one_item[1].replace("_", " ") + "<SEGMENTER>" + one_item[0])
        print('[Info] {} instances from {} {} test set'.format(len(one_list), data_path, one_type))

    tmp_train_list = data_processing_tweet_Covid(tmp_train_list)
    tmp_dev_list = data_processing_tweet_Covid(tmp_dev_list)
    tmp_test_list = data_processing_tweet_Covid(tmp_test_list)

    if only_binary_polarity is True:
        tmp_train_list = [i for i in tmp_train_list if i[0] < 2]
        tmp_dev_list = [i for i in tmp_dev_list if i[0] < 2]
        tmp_test_list = [i for i in tmp_test_list if i[0] < 2]

    for i in range(3):
        print(tmp_train_list[i])

    return tmp_train_list, tmp_dev_list, tmp_test_list


def data_reading_Adversarial_Samples(data_path):
    lemmatizer = WordNetLemmatizer()
    tmp_sample_list = []
    tmp_list = open(data_path, encoding="utf-8").readlines()
    for tmp_i in tmp_list:
        tmp_label = int(tmp_i.split("<segmenter>")[0].strip())
        tmp_content = tmp_i.split("<segmenter>")[1].strip()

        tmp_target = tmp_content[:tmp_content.index("<s>")]
        tmp_target = " ".join([lemmatizer.lemmatize(i) for i in tmp_target.split()])
        tmp_target = re.sub("[^a-zA-Z0-9\']", " ", tmp_target)
        tmp_target = re.sub("\s+", " ", tmp_target).strip()

        if len(tmp_target) > 2:
            tmp_text = tmp_content[tmp_content.index("<s>") + 3:].split()
            tmp_text = " ".join([normalizeToken(i) for i in tmp_text])
            tmp_text = re.sub("\s+", " ", tmp_text)
            tmp_sample_list.append([tmp_label, tmp_target + " <s> " + tmp_text])

    if only_binary_polarity is True:
        tmp_sample_list = [i for i in tmp_sample_list if i[0] < 2]

    print('[Info] {} instances from {} train set'.format(len(tmp_sample_list), data_path))

    for i in range(3):
        print(tmp_sample_list[i])

    return tmp_sample_list


def data_reading_sample_from_txt(data_path):
    tmp_sample_list = []
    tmp_list = open(data_path, encoding="utf-8").readlines()
    for tmp_i in tmp_list:
        tmp_label = int(tmp_i.split("<segmenter>")[0].strip())
        tmp_content = tmp_i.split("<segmenter>")[1].strip()

        tmp_target = tmp_content[:tmp_content.index("<s>")]
        tmp_target = re.sub("\s+", " ", tmp_target).strip()

        if len(tmp_target) > 2:
            tmp_text = tmp_content[tmp_content.index("<s>") + 3:].split()
            # tmp_text = " ".join([normalizeToken(i) for i in tmp_text])
            tmp_text = " ".join(tmp_text)
            tmp_text = re.sub("\s+", " ", tmp_text)
            tmp_sample_list.append([tmp_label, tmp_target + " <s> " + tmp_text])

    if only_binary_polarity is True:
        tmp_sample_list = [i for i in tmp_sample_list if i[0] < 2]

    print('[Info] {} instances from {} train set'.format(len(tmp_sample_list), data_path))

    for i in range(3):
        print(tmp_sample_list[i])

    return tmp_sample_list
