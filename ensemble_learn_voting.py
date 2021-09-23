import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertweetTokenizer, AutoModel, AutoTokenizer, AdamW,\
    get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
# from vlad.mynextvlad import NeXtVLAD
# from vlad.internetnextvlad import NextVLAD

torch.manual_seed(7)
np.random.seed(7)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


TRAIN_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/train4000_shuf.jsonl"
VAL_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/val1000_shuf.jsonl"
TEST_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/twitter_test_wl_shuf.jsonl"
SAVE_PATH = "/s/lovelace/c/nobackup/iray/sinamps/tempmodels/"
# LOAD_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/model/next/nedsdsds"

PRE_TRAINED_BERT = 'bert-large-cased'
PRE_TRAINED_ROBERTA = 'roberta-large'
PRE_TRAINED_CTBERT = 'digitalepidemiologylab/covid-twitter-bert-v2'
# PRE_TRAINED_BERTWEET = 'vinai/bertweet-base'

MAXTOKENS = 512
NUM_EPOCHS = 8
EMB = 1024
H = 512
BS = 4
INITIAL_LR = 1e-6
WARMUP = 500
save_epochs = [3, 5]

CUDA_0 = 'cuda:0'
CUDA_1 = 'cuda:1'
CUDA_2 = 'cuda:2'


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


def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


def load_data(file_name):
    # Load and prepare the data
    texts = []
    labels = []
    try:
        f = open(file_name, encoding='utf8')
    except:
        print('my log: could not read file')
        exit()
    lines = f.readlines()
    # dataset = ensemble_data(lines, 3)
    for line in lines:
        data = json.loads(line)
        resp = data['response']
        contexts = ''
        # reduced_context = data['context'][-3:]
        for context in data['context'][-3:]:
            contexts = contexts + ' [SEP] ' + context
        contexts = contexts[7:]
        contexts = contexts + ' [SEP] '
        thread = contexts + resp
        # cleaned_sent = tp.clean(sent)
        texts.append(thread)
        labels.append(1 if (data["label"] == "SARCASM") else 0)
    return texts, labels


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, bert_encodings, roberta_encodings, ctbert_encodings, labels):
        self.bert_encodings = bert_encodings
        self.roberta_encodings = roberta_encodings
        self.ctbert_encodings = ctbert_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {('bert_' + key): torch.tensor(val[idx]) for key, val in self.bert_encodings.items()}
        item2 = {('roberta_' + key): torch.tensor(val[idx]) for key, val in self.roberta_encodings.items()}
        item3 = {('ctbert_' + key): torch.tensor(val[idx]) for key, val in self.ctbert_encodings.items()}
        item.update(item2)
        item.update(item3)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)


def feed_model(model, data_loader):
    outputs_flat = []
    labels_flat = []
    for batch in data_loader:
        bert_input_ids = batch['bert_input_ids'].to(CUDA_0)
        roberta_input_ids = batch['roberta_input_ids'].to(CUDA_0)
        ctbert_input_ids = batch['ctbert_input_ids'].to(CUDA_0)
        bert_attention_mask = batch['bert_attention_mask'].to(CUDA_0)
        roberta_attention_mask = batch['roberta_attention_mask'].to(CUDA_0)
        ctbert_attention_mask = batch['ctbert_attention_mask'].to(CUDA_0)
        outputs = model(bert_input_ids, roberta_input_ids, ctbert_input_ids,
                        bert_attention_mask=bert_attention_mask,
                        roberta_attention_mask=roberta_attention_mask,
                        ctbert_attention_mask=ctbert_attention_mask
                        )
        outputs = outputs.detach().cpu().numpy()
        labels = batch['labels'].to('cpu').numpy()
        outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
        labels_flat.extend(labels.flatten())
        del outputs, labels, bert_input_ids, bert_attention_mask, roberta_input_ids, roberta_attention_mask, \
            ctbert_input_ids, ctbert_attention_mask
    return labels_flat, outputs_flat


class EnsembleModel(nn.Module):
    def __init__(self, bert_model, roberta_model, ctbert_model, n_classes, dropout=0.05):
        super().__init__()

        self.bert_model = bert_model.to(CUDA_0)
        self.roberta_model = roberta_model.to(CUDA_1)
        self.ctbert_model = ctbert_model.to(CUDA_2)
        self.bert_final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(EMB, EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(EMB, n_classes)
        ).to(CUDA_0)
        self.roberta_final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(EMB, EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(EMB, n_classes)
        ).to(CUDA_1)
        self.ctbert_final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(EMB, EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(EMB, n_classes)
        ).to(CUDA_2)

        self.final = nn.Sequential(
            nn.Linear(6, n_classes)
        ).to(CUDA_2)

    def forward(self, bert_input_, roberta_input_, ctbert_input_, **kwargs):
        X1 = bert_input_
        X2 = roberta_input_
        X3 = ctbert_input_
        if 'bert_attention_mask' in kwargs:
            bert_attention_mask = kwargs['bert_attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        if 'roberta_attention_mask' in kwargs:
            roberta_attention_mask = kwargs['roberta_attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        if 'ctbert_attention_mask' in kwargs:
            ctbert_attention_mask = kwargs['ctbert_attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        hidden_states_bert = self.bert_model(X1.to(CUDA_0),
                                             attention_mask=bert_attention_mask.to(CUDA_0)).last_hidden_state
        hidden_states_roberta = self.roberta_model(X2.to(CUDA_1),
                                                   attention_mask=roberta_attention_mask.to(CUDA_1)).last_hidden_state
        hidden_states_ctbert = self.ctbert_model(X3.to(CUDA_2),
                                                     attention_mask=ctbert_attention_mask.to(CUDA_2)).last_hidden_state
        cls1 = hidden_states_bert[:, 0, :]
        cls2 = hidden_states_roberta[:, 0, :]
        cls3 = hidden_states_ctbert[:, 0, :]
        myo1 = self.bert_final(cls1.to(CUDA_0))
        myo2 = self.roberta_final(cls2.to(CUDA_1))
        myo3 = self.ctbert_final(cls3.to(CUDA_2))
        myo = torch.cat((myo1.to(CUDA_2), myo2.to(CUDA_2), myo3.to(CUDA_2)), 1)
        myo = self.final(myo)
        # myo = nn.functional.softmax(myo, dim=1)
        return myo


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w', encoding='utf8')
    myprint(INITIAL_LR, logfile)
    train_texts, train_labels = load_data(TRAIN_PATH)
    val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_BERT)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_ROBERTA)
    ctbert_tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_CTBERT)
    bert_train_encodings = bert_tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    bert_val_encodings = bert_tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    bert_test_encodings = bert_tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    roberta_train_encodings = roberta_tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    roberta_val_encodings = roberta_tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    roberta_test_encodings = roberta_tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    ctbert_train_encodings = ctbert_tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    ctbert_val_encodings = ctbert_tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    ctbert_test_encodings = ctbert_tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)

    train_dataset = MyDataset(bert_train_encodings, roberta_train_encodings, ctbert_train_encodings, train_labels)
    val_dataset = MyDataset(bert_val_encodings, roberta_val_encodings, ctbert_val_encodings, val_labels)
    test_dataset = MyDataset(bert_test_encodings, roberta_test_encodings, ctbert_test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bert_model = BertModel.from_pretrained(PRE_TRAINED_BERT)
    roberta_model = RobertaModel.from_pretrained(PRE_TRAINED_ROBERTA)
    ctbert_model = AutoModel.from_pretrained(PRE_TRAINED_CTBERT)
    model = EnsembleModel(bert_model=bert_model, roberta_model=roberta_model, ctbert_model=ctbert_model, n_classes=2)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    optim = AdamW(model.parameters(), lr=INITIAL_LR)
    total_steps = len(train_loader)/BS * epochs
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=WARMUP,
                                                num_training_steps=total_steps)
    loss_model = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch+1, epochs))
        outputs_flat = []
        labels_flat = []
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optim.zero_grad()
            bert_input_ids = batch['bert_input_ids'].to(CUDA_0)
            bert_attention_mask = batch['bert_attention_mask'].to(CUDA_0)
            roberta_input_ids = batch['roberta_input_ids'].to(CUDA_1)
            roberta_attention_mask = batch['roberta_attention_mask'].to(CUDA_1)
            ctbert_input_ids = batch['ctbert_input_ids'].to(CUDA_2)
            ctbert_attention_mask = batch['ctbert_attention_mask'].to(CUDA_2)
            labels = batch['labels'].to(CUDA_2)
            outputs = model(bert_input_ids, roberta_input_ids, ctbert_input_ids,
                            bert_attention_mask=bert_attention_mask,
                            roberta_attention_mask=roberta_attention_mask,
                            ctbert_attention_mask=ctbert_attention_mask,
                            labels=labels)
            loss = loss_model(outputs, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
            labels_flat.extend(labels.flatten())
            del outputs, labels, bert_attention_mask, roberta_attention_mask, ctbert_attention_mask,\
                bert_input_ids, roberta_input_ids, ctbert_input_ids
        evaluate_model(labels_flat, outputs_flat, 'Train set Result epoch ' + str(epoch+1), logfile)
        del labels_flat, outputs_flat
        model.eval()
        val_labels, val_predictions = feed_model(model, val_loader)
        evaluate_model(val_labels, val_predictions, 'Validation set Result epoch ' + str(epoch+1), logfile)
        del val_labels, val_predictions
        if (epoch+1) in save_epochs:
            test_labels, test_predictions = feed_model(model, test_loader)
            evaluate_model(test_labels, test_predictions, 'Test set Result epoch ' + str(epoch+1), logfile)
            del test_labels, test_predictions
            try:
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-auto-' + str(epoch+1)))
            except:
                myprint("Could not save the model", logfile)
        model.train()
    del train_loader, val_loader
    model.eval()
    myprint('--------------Training complete--------------', logfile)
    torch.save(model.state_dict(), SAVE_PATH + args[0].split('/')[-1][:-3] + '-final')
    test_labels, test_predictions = feed_model(model, test_loader)
    evaluate_model(test_labels, test_predictions, 'Final Testing', logfile)
    del test_labels, test_predictions
