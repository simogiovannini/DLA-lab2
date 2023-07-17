import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, DistilBertModel
from datasets import load_dataset
from utils.race_wrapper import RaceWrapper
from models.ranking_predictor import RankingPredictor
import os.path
from tqdm.auto import tqdm
import math


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x_1, x_2, target):
        return torch.div(x_1, x_2).sum()


def compute_accuracy(logits):
    correct_count = 0
    for i in range(0, len(logits), 4):
        m = max(logits[i].item(), logits[i + 1].item(), logits[i + 2].item(), logits[i + 3].item())
        if math.isclose(m, logits[i].item()):
            correct_count += 1
    return correct_count * 100 / (len(logits)/4)


def compute_loss(loss, logits, device):
    x_1 = torch.empty(0).to(device)
    x_2 = torch.empty(0).to(device)
    target = torch.empty(0).to(device)

    for i in range(logits.shape[0]):
        if i % 4 == 0:
            x_1 = torch.cat((x_1, logits[i], logits[i], logits[i]))
        else:
            x_2 = torch.cat((x_2, logits[i]))
            target = torch.cat((target, torch.tensor(1).to(device).reshape(1)))
    return loss(x_1, x_2, target)


def compute_features(d, b_size, n, idx, device):
    start = idx * b_size
    end = min(idx * b_size + b_size, n)

    articles_batch = d['article'][start: end]
    questions_batch = d['question'][start: end]
    options_batch = d['options'][start: end]
    answers_batch = d['answer'][start: end]

    features = torch.empty(0, 768).to(device)

    for i in range(articles_batch.shape[0]):
        a = articles_batch[i]
        q = questions_batch[i]
        correct = int(answers_batch[i].item())
        tmp = torch.empty(0, 768).to(device)
        for k in range(len(options_batch[i])):
            curr_opt = options_batch[i][k]
            opt_with_context = torch.add(a, q).add(curr_opt)
            opt_with_context = torch.div(opt_with_context, 3)
            if k == correct:
                tmp = torch.cat((opt_with_context[None, :], tmp), 0)
            else:
                tmp = torch.cat((tmp, opt_with_context[None, :]), 0)
        features = torch.cat((features, tmp), 0)
    return features, answers_batch


def permute_dataset(d):
    indices = torch.randperm(d['question'].size()[0])
    d['article'] = d['article'][indices]
    d['question'] = d['question'][indices]
    d['options'] = d['options'][indices]
    d['answer'] = d['answer'][indices]
    return d


def get_cls_embedding(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0][None, :]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('runs/exercise-3_2')


path = 'datasets/race_embedded_train'

if not os.path.isfile(path):
    dataset = load_dataset('race', 'all')

    dataset = {
        'train': RaceWrapper(dataset['train']),
        'validation': RaceWrapper(dataset['validation']),
        'test': RaceWrapper(dataset['test'])
    }

    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    split_to_index = {'train': 0, 'validation': 1, 'test': 2}

    for split in dataset.keys():

        article_embeddings = torch.empty(0, 768).to(device)
        options_embeddings = torch.empty(0, 4, 768).to(device)
        question_embeddings = torch.empty(0, 768).to(device)
        answers = torch.empty(0).to(device)

        n = len(dataset[split])
        for i in range(n):
            print(f'{split} - {i}/{n}')
            ex_id, article, options, question, answer = dataset[split][i]

            opt_list = torch.empty(0, 768).to(device)
            for opt in options:
                opt_list = torch.cat((opt_list, get_cls_embedding(bert, bert_tokenizer, opt).to(device)))
            options_embeddings = torch.cat((options_embeddings, opt_list[None, :, :]), 0)
            del opt_list

            article_embeddings = torch.cat(
                (article_embeddings, get_cls_embedding(bert, bert_tokenizer, article).to(device)), 0)
            question_embeddings = torch.cat(
                (question_embeddings, get_cls_embedding(bert, bert_tokenizer, question).to(device)), 0)
            answers = torch.cat((answers, torch.tensor(letter_to_number[answer]).reshape(1).to(device)), 0)

            torch.cuda.empty_cache()

        data_to_save = {
            'article': article_embeddings,
            'question': question_embeddings,
            'options': options_embeddings,
            'answer': answers
        }

        torch.save(data_to_save, f'datasets/race_embedded_{split}')


print('Loading datasets')
train_dataset = torch.load('datasets/race_embedded_train', map_location=torch.device('cuda'))
val_dataset = torch.load('datasets/race_embedded_validation', map_location=torch.device('cuda'))
test_dataset = torch.load('datasets/race_embedded_test', map_location=torch.device('cuda'))
print('Loading complete')

ranker = RankingPredictor(input_shape=768, hidden_units_1=512, hidden_units_2=256, hidden_units_3=128)
ranker.to(device)

batch_size = 128
num_epochs = 5
lr = 0.1

loss_fn = nn.MarginRankingLoss(margin=1)
# loss_fn = CustomLoss()
optimizer = torch.optim.SGD(ranker.parameters(), lr=lr)

n_train_samples = len(train_dataset['question'])
n_val_samples = len(val_dataset['question'])
n_test_samples = len(test_dataset['question'])

n_train_batches = int(n_train_samples / batch_size) if n_train_samples % batch_size == 0 else int(
    n_train_samples / batch_size) + 1
n_val_batches = int(n_val_samples / batch_size) if n_val_samples % batch_size == 0 else int(
    n_val_samples / batch_size) + 1
n_test_batches = int(n_test_samples / batch_size) if n_test_samples % batch_size == 0 else int(
    n_test_samples / batch_size) + 1

for epoch in tqdm(range(num_epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_dataset = permute_dataset(train_dataset)

    ### Training
    train_loss = 0
    for i in range(n_train_batches):
        # print(f'Epoch: {epoch} ------- {i + 1}/{n_train_batches} batches')

        ranker.train()

        X, _ = compute_features(train_dataset, batch_size, n_train_samples, i, device)

        y_logits = ranker(X).to(device)
        loss = compute_loss(loss_fn, y_logits, device)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= n_train_batches

    ### Validation
    val_loss, val_acc = 0, 0
    ranker.eval()
    with torch.inference_mode():
        for i in range(n_val_batches):
            X, y = compute_features(val_dataset, batch_size, n_val_samples, i, device)

            val_logits = ranker(X).to(device)

            val_loss += compute_loss(loss_fn, val_logits, device)

            val_acc += compute_accuracy(val_logits)

        val_loss /= n_val_batches
        val_acc /= n_val_batches

    print(f"\nTrain loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}, Validation acc: {val_acc:.2f}%\n")
    writer.add_scalar(f'QA Ranker/Loss/Train', train_loss, epoch)
    writer.add_scalar(f'QA Ranker/Loss/Test', val_loss, epoch)
    writer.add_scalar(f'QA Ranker/Accuracy/Test', val_acc, epoch)

torch.save(ranker.state_dict(), 'trained_models/3_2-ranker')
