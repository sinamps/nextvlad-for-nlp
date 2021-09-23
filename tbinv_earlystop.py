# Author: Sina Mahdipour Saravani
# Link to our paper for this project:
# https://sinamps.github.io/publication/2021-08-05-emnlp-negative-results
import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import time
# from vlad.mynextvlad import NeXtVLAD  # this imports our implementation of the NeXtVLAD layer
from vlad.internetnextvlad import NextVLAD  # this imports an implementation of the NeXtVLAD layer taken from
# https://www.kaggle.com/gibalegg/mtcnn-nextvlad#NextVLAD

# Setting manual seed for various libs for reproducibility purposes.
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)
# Setting PyTorch's required configuration variables for reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# To run the code in a reproducibale way, use the following running parameter for CUDA 10.2 or higher:
# CUBLAS_WORKSPACE_CONFIG=:16:8 python tbinv_earlystop.py
# If you do not care about reproducibility, you can comment above configs and run the script without the parameter

# Path specification for Train, Validation, and Test sets. Make sure you edit them properly to point to your local
# datasets.
TRAIN_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/train4000_shuf.jsonl"
# TRAIN_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
VAL_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/val1000_shuf.jsonl"
# VAL_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
TEST_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/twitter_test_wl_shuf.jsonl"
# TEST_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
SAVE_PATH = "/s/lovelace/c/nobackup/iray/sinamps/tempmodels/"
# If you want to load an already fine-tuned model and continue its training, uncomment and edit the following line.
# LOAD_PATH = "/s/lovelace/c/nobackup/iray/sinamps/claim_project/2021-06-01_models/jun10_tbinv_ensemble_with_bigger_batch-final-epoch-15"

# Configuration variables to choose the pre-trained model you want to use and other training settings:
# the pre-trained model name from huggingface transformers library names:
PRE_TRAINED_MODEL = 'bert-large-cased'
# it can be from the followings for example: 'digitalepidemiologylab/covid-twitter-bert-v2',
#                                            'bert-large-uncased',
#                                            'vinai/bertweet-base'
#                                            'xlnet-base-cased'

MAXTOKENS = 512
NUM_EPOCHS = 50  # default maximum number of epochs
BERT_EMB = 1024  # set to either 768 or 1024 for BERT-Base and BERT-Large models respectively
BS = 4  # batch size
INITIAL_LR = 1e-6  # initial learning rate
save_epochs = [1, 2, 3, 4, 5, 6, 7]  # these are the epoch numbers (starting from 1) to test the model on the test set
# and save the model checkpoint.
EARLY_STOP_PATIENCE = 2  # If model does not improve for this number of epochs, training stops.

# Setting GPU cards to use for training the model. Make sure you read our paper to figure out if you have enough GPU
# memory. If not, you can change all of them to 'cpu' to use CPU instead of GPU. By the way, two 24 GB GPU cards are
# enough for current configuration, but in case of developing based on this you may need more (that's why there are
# three cards declared here)
CUDA_0 = 'cuda:2'
CUDA_1 = 'cuda:3'
CUDA_2 = 'cuda:3'

# The function for applying the data expansion technique (DE) mentioned in our paper.
def ensemble_data(dataset, context_size=3):
    new_dataset = []
    for line in dataset:
        data = json.loads(line)
        if len(data['context']) > context_size:
            data['context'] = data['context'][-context_size:]

        for i in range(min(context_size, len(data['context']))):
            d = {'label': data['label'],
                 'response': data['response'],
                 'context': data['context'][i:]}
            new_dataset.append(d)
    return new_dataset

# The function for printing in both console and a given log file.
def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


# The function for loading datasets from jsonl files and returning them in lists. This function also concatenates the
# three most recent context tweets with the response tweet and ignores the rest of contexts.
def load_data(file_name):
    texts = []
    labels = []
    try:
        f = open(file_name)
    except:
        print('my log: could not read file')
        exit()
    lines = f.readlines()
    # dataset = ensemble_data(lines, 3)  # if you want to apply the data expansion (DE from our paper), uncomment this.
    for line in lines:
        data = json.loads(line)
        resp = data['response']
        contexts = ''
        # To make sure that with even lower values of MAXTOKENS variable, we do not lose the response tweet, I put it
        # at the beginning of the text.
        for context in data['context'][-3:]:
            contexts = context + ' [SEP] ' + contexts
        contexts = contexts[:-7]
        contexts = ' [SEP] ' + contexts
        thread = resp + contexts
        # cleaned_sent = tp.clean(sent)  # I initially used the tweet processor library for cleaning text, but ignored
        # it later. The lib: https://pypi.org/project/tweet-preprocessor/
        texts.append(thread)
        labels.append(1 if (data["label"] == "SARCASM") else 0)
    return texts, labels


# Overriding the Dataset class required for the use of PyTorch's data loader classes.
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# The function to compute and print the performance measure scores using sklearn implementations.
def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)
    return f1_score(labels, predictions)  # return f-1 for positive class (sarcasm) as the early stopping measure.


# The function to do a forward pass of the network.
def feed_model(model, data_loader):
    outputs_flat = []
    labels_flat = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(CUDA_0)  # 'input_ids' are the index of tokens in model's dict generated by
        # transformer tokenizer
        attention_mask = batch['attention_mask'].to(CUDA_0)
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.detach().cpu().numpy()
        labels = batch['labels'].to('cpu').numpy()
        outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
        labels_flat.extend(labels.flatten())
        del outputs, labels, attention_mask, input_ids
    return labels_flat, outputs_flat


class MyModel(nn.Module):
    # Each component other than the Transformer, are in a sequential layer (it is not required obviously, but it is
    # possible to stack them with other layers if desired)
    def __init__(self, base_model, n_classes, dropout=0.05):
        super().__init__()

        self.base_model = base_model.to(CUDA_0)

        self.mylstm = nn.Sequential(
            nn.LSTM(input_size=BERT_EMB, hidden_size=BERT_EMB, num_layers=2, dropout=0.25, batch_first=True,
                    bidirectional=True)
        ).to(CUDA_1)

        self.myvlad = nn.Sequential(
            # You can change NeXtVLAD's configuration by changing these input parameters. Note that they are a bit
            # different in this implementation and the one we wrote ourselves.
            NextVLAD(num_clusters=128, dim=BERT_EMB, expansion=4, num_class=n_classes)
        ).to(CUDA_2)

    def forward(self, input_, **kwargs):
        X = input_
        if 'attention_mask' in kwargs:
            attention_mask = kwargs['attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        hidden_states = self.base_model(X.to(CUDA_0), attention_mask=attention_mask.to(CUDA_0)).last_hidden_state
        bert_tokens = hidden_states[:, :, :]  # here we use all tokens, in other cases one may only want the CLS token
        lstm_out, (hn, cn) = self.mylstm(bert_tokens.to(CUDA_1))
        lstm_out = nn.functional.leaky_relu(lstm_out[:, :, :BERT_EMB] + lstm_out[:, :, BERT_EMB:])  # sum LSTM' two
        # directions
        vlad_out = self.myvlad(lstm_out.to(CUDA_2))
        # myo = nn.functional.softmax(vlad_out, dim=1)  # softmax is not required as we are using CrossEntropy loss
        # that applies the softmax itself.
        my_output = vlad_out
        return my_output


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    myprint("Please wait for the model to download and load sub-models, getting a few warnings is OK.", logfile)

    train_texts, train_labels = load_data(TRAIN_PATH)
    val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    tokenizer.model_max_length = MAXTOKENS
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)
    test_dataset = MyDataset(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    model = MyModel(base_model=base_model, n_classes=2)
    # If you want to load an already fine-tuned model and continue its training, uncomment the next line.
    # model.load_state_dict(torch.load(LOAD_PATH))
    model.train()  # take model to training mode

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=False)  # shuffle False for reproducibility
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    optim = AdamW(model.parameters(), lr=INITIAL_LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=2000,
                                                num_training_steps=total_steps)
    loss_model = nn.CrossEntropyLoss()
    best_val_f1 = None
    patience_counter = 0  # counter to check if patience for early stopping has been reached

    # Training loop:
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch+1, epochs))
        outputs_flat = []
        labels_flat = []
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optim.zero_grad()
            input_ids = batch['input_ids'].to(CUDA_0)  # 'input_ids' are the index of tokens in model's dictionary
            attention_mask = batch['attention_mask'].to(CUDA_0)  # 'attention_mask' indicates the non-padding tokens
            labels = batch['labels'].to(CUDA_2)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_model(outputs, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
            labels_flat.extend(labels.flatten())
            del outputs, labels, attention_mask, input_ids
        evaluate_model(labels_flat, outputs_flat, 'Train set Result epoch ' + str(epoch+1), logfile)
        del labels_flat, outputs_flat
        model.eval()
        val_labels, val_predictions = feed_model(model, val_loader)
        val_f1 = evaluate_model(val_labels, val_predictions, 'Validation set Result epoch ' + str(epoch+1), logfile)
        del val_labels, val_predictions
        myprint("------------------------------- Val F1 at epoch " + str(epoch+1) + " : " + str(val_f1), logfile)
        # The early stopping logic:
        if best_val_f1 is None:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
        elif val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            myprint("Better Val F-1 score; saving Model", logfile)
            patience_counter = 0
            torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
        else:
            patience_counter = patience_counter + 1
            myprint("Worse Val F-1 score; Patience Counter:" + str(patience_counter), logfile)
            if patience_counter >= EARLY_STOP_PATIENCE:  # patience reached, stop training
                # Load the last best model:
                model.load_state_dict(torch.load(SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
                # Test the model on the testing set:
                test_labels, test_predictions = feed_model(model, test_loader)
                evaluate_model(test_labels, test_predictions, 'Test set Result epoch ' + str(epoch + 1), logfile)
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-EarlyStoppedFinal-' +
                                                'time:' + str(time.time())))
                break
        model.train()
    del train_loader, val_loader, test_loader
    # End of main

