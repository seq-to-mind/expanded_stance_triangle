import os
import sys
import time
import random
import re
import pickle

import json
import math
import numpy as np
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import AutoModel, AutoTokenizer


import global_config
from corpus_reading import data_reading_sample_from_txt, data_reading_Adversarial_Samples

device = 'cuda:' + global_config.gpu_id if cuda.is_available() else 'cpu'

running_random_number = random.randint(1000, 9999)
print("running_random_number", running_random_number, "\n")

global_label_dict = {"AGAINST": 0, "FAVOR": 1, "NONE": 2, "Neutral": 3}
global_label_dict_rev = {0: "AGAINST", 1: "FAVOR", 2: "NONE", 3: "Neutral"}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])
    return batches


class NeuralClassifier(nn.Module):
    def __init__(self, model_type, output_attn=False):
        super(NeuralClassifier, self).__init__()
        print("Loading the language backbone", model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False)
        self.language_backbone = AutoModel.from_pretrained(model_type, output_hidden_states=False, output_attentions=output_attn)

        self.drop_out = nn.Dropout(p=0.3)
        self.linear = nn.Linear(768, 3)

    def forward(self, batch):
        x_batch = [i[1] for i in batch]
        x_batch = self.tokenizer(x_batch, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True, max_length=global_config.max_input_length).data
        hidden_state = self.language_backbone(x_batch["input_ids"].to(device), attention_mask=x_batch["attention_mask"].to(device))[0][:, 0, :]
        return_logits = self.linear(self.drop_out(hidden_state))
        return return_logits


class ClassificationAgent:
    def __init__(self):

        model_type = global_config.pretrained_model

        self.model = NeuralClassifier(model_type)
        self.model.to(device).train()

        print('[Info] Built a model with {} parameters'.format(sum(p.numel() for p in self.model.parameters())))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=global_config.learning_rate, weight_decay=5e-5)
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size = global_config.batch_size
        self.epoch_num = global_config.training_epoch_num
        self.random_seed = global_config.random_seed

    def testing(self, epoch_i, test_set_dict):
        output_fp = open("running_log_" + str(running_random_number) + '.txt', "a+")
        for one_k in test_set_dict.keys():
            one_test_set = test_set_dict[one_k]
            self.model.eval()
            total_num = 0.
            total_loss = 0.

            if global_config.check_test_result is True:
                preview_file_fp = open("_".join([".test_result_preview", one_k, str(epoch_i)]), "w", encoding="utf-8")

            all_pred_polarity, all_gold_polarity = [], []
            all_entropy = []

            with torch.no_grad():
                test_batches = get_batches(one_test_set, self.batch_size)
                for batch in test_batches:
                    y_polarity_gold = torch.tensor([i[0] for i in batch]).to(device)
                    y_polarity_logits = self.model(batch)
                    loss_polarity = self.loss_function(y_polarity_logits, y_polarity_gold)
                    loss = loss_polarity
                    total_loss += loss

                    _, y_polarity_pred = torch.max(y_polarity_logits, dim=-1)
                    y_polarity_prob = F.softmax(y_polarity_logits, dim=-1).cpu().numpy().tolist()
                    all_entropy.extend([entropy(i) for i in y_polarity_prob])

                    y_polarity_pred = y_polarity_pred.detach().cpu().numpy().tolist()
                    all_pred_polarity.extend(y_polarity_pred)
                    y_polarity_gold = [i[0] for i in batch]
                    all_gold_polarity.extend([i[0] for i in batch])

                    total_num += len(y_polarity_pred)

                    if global_config.check_test_result is True:
                        for k, v in enumerate(y_polarity_pred):
                            if y_polarity_pred[k] != y_polarity_gold[k]:
                                preview_file_fp.write("Target: " + global_label_dict_rev[y_polarity_gold[k]] + \
                                                      " | Polarity Pred: " + global_label_dict_rev[y_polarity_pred[k]] + \
                                                      " " + str(y_polarity_prob[k]) + "\n")
                                preview_file_fp.write(batch[k][1] + "\n\n")

            if global_config.check_test_result is True:
                preview_file_fp.close()

            total_f1 = f1_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro")
            total_precision = precision_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro")
            total_recall = recall_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro")
            print('[Info] Epoch {:02d} 3-class test: {} | F1 {:.4f}% | P {:.4f}% | R {:.4f}% | loss {:.4f}'
                  .format(epoch_i, one_k, total_f1, total_precision, total_recall, total_loss / total_num))

            total_f1 = f1_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro", labels=[0, 1])
            total_precision = precision_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro", labels=[0, 1])
            total_recall = recall_score(y_pred=all_pred_polarity, y_true=all_gold_polarity, average="macro", labels=[0, 1])
            print('[Info] Epoch {:02d} 2-class test: {} | F1 {:.4f}% | P {:.4f}% | R {:.4f}% | loss {:.4f}'
                  .format(epoch_i, one_k, total_f1, total_precision, total_recall, total_loss / total_num))

            output_fp.write("\n*************** Epoch{} {} ***************\n".format(epoch_i, one_k))
            output_fp.write(classification_report(y_pred=all_pred_polarity, y_true=all_gold_polarity, digits=3))

            if epoch_i == -999:
                print(one_k, "entropy", str(np.mean(all_entropy)))

        output_fp.close()

    def training(self, train_set, test_set_dict):
        for epoch_i in range(self.epoch_num):
            self.model.train()

            random.seed(global_config.random_seed + epoch_i)
            random.shuffle(train_set)
            train_batches = get_batches(train_set, self.batch_size)

            scheduled_lr = global_config.learning_rate * (0.9 ** epoch_i)
            print("[Info] learning rate is changed to:", scheduled_lr)
            self.optimizer.param_groups[0]["lr"] = scheduled_lr

            for idx, batch in enumerate(train_batches):
                y_polarity_gold = torch.tensor([i[0] for i in batch]).to(device)

                self.optimizer.zero_grad()
                self.model.zero_grad()
                y_polarity_logits = self.model(batch)
                loss_polarity = self.loss_function(y_polarity_logits, y_polarity_gold)
                loss = loss_polarity
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                loss.backward()
                self.optimizer.step()

                if idx % int(len(train_batches) / 2) == 0 or idx == len(train_batches) - 1:
                    print("Epoch", epoch_i, "Iter", idx, "Training loss:", loss.item())

            with torch.no_grad():
                self.testing(epoch_i, test_set_dict)
                if epoch_i > 0 and global_config.save_checkpoint:
                    save_path = 'saved_models/model_' + str(running_random_number) + "_" + str(epoch_i) + '.chkpt'
                    torch.save(self.model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated')


def main():
    setup_seed(global_config.random_seed)

    train_set_tweet_stance_A = data_reading_sample_from_txt("data/SemEval16_Tweet_Stance_Detection/Original_Annotation/SemEval16_TweetTask_A_Train.txt")
    test_set_tweet_stance_A = data_reading_sample_from_txt("data/SemEval16_Tweet_Stance_Detection/Original_Annotation/SemEval16_TweetTask_A_Test.txt")
    test_set_tweet_stance_B = data_reading_sample_from_txt("data/SemEval16_Tweet_Stance_Detection/Original_Annotation/SemEval16_TweetTask_B_Test.txt")
    train_adversarial_set_tweet_stance = (data_reading_Adversarial_Samples("data/SemEval16_Tweet_Stance_Detection/Enriched_Annotation/SemEval16_TweetTask_A_Adversarial_Set.txt") +
                                          data_reading_Adversarial_Samples("data/SemEval16_Tweet_Stance_Detection/Enriched_Annotation/SemEval16_TweetTask_A_Contrastive_Neutral_Set.txt"))

    merged_train_set = train_set_tweet_stance_A + train_adversarial_set_tweet_stance
    # merged_train_set =  train_adversarial_set_tweet_stance
    random.shuffle(merged_train_set)

    merged_test_set_dict = {"Tweet_Stance_A": test_set_tweet_stance_A, "Tweet_Stance_B": test_set_tweet_stance_B}

    print("[Info] Train set size:", len(merged_train_set))
    for k in merged_test_set_dict.keys():
        print("[Info]", k, "test set size:", len(merged_test_set_dict[k]))

    agent = ClassificationAgent()
    if global_config.train_mode is True:
        agent.training(merged_train_set, merged_test_set_dict)
    else:
        print("Loading pre-trained checkpoint from:", global_config.checkpoint_to_load)
        agent.model.load_state_dict(torch.load(global_config.checkpoint_to_load, map_location=torch.device('cuda:' + global_config.gpu_id)))
        agent.testing(-999, merged_test_set_dict)


if __name__ == '__main__':
    main()
