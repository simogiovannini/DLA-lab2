import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from models.text_classifier import TextClassifier
from tqdm.auto import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('runs/exercise-3_1')

dataset = load_dataset('ag_news')

train_labels = torch.tensor(dataset['train']['label'])
test_labels = torch.tensor(dataset['test']['label'])
del dataset

train_features = torch.load('datasets/ag_news_train_post_gpt')
test_features = torch.load('datasets/ag_news_test_post_gpt')

classifier = TextClassifier(input_shape=768, hidden_units_1=512, hidden_units_2=256, hidden_units_3=128, output_shape=4)
classifier.to(device)

batch_size = 512
num_epochs = 100
lr = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)

n_train_samples = len(train_features)
n_test_samples = len(test_features)

n_train_batches = int(n_train_samples / batch_size) if n_train_samples % batch_size == 0 else int(n_train_samples / batch_size) + 1
n_test_batches = int(n_test_samples / batch_size) if n_test_samples % batch_size == 0 else int(n_test_samples / batch_size) + 1

for epoch in tqdm(range(num_epochs)):
    print(f"Epoch: {epoch}\n-------")

    indices = torch.randperm(train_features.size()[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    ### Training
    train_loss = 0
    for i in range(n_train_batches):
        #print(f'Epoch: {epoch} ------- {i + 1}/{n_train_batches} batches')

        classifier.train()

        X = train_features[i * batch_size: min(i * batch_size + batch_size, n_train_samples)]
        y = train_labels[i * batch_size: min(i * batch_size + batch_size, n_train_samples)]

        y_logits = classifier(X.to(device))
        y_probs = torch.softmax(y_logits, dim=1)
        y_pred = y_probs.argmax(dim=1)

        loss = loss_fn(y_logits.to(device), y.to(device))
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= n_train_batches

    ### Testing
    test_loss, test_acc = 0, 0
    classifier.eval()
    with torch.inference_mode():
        for i in range(n_test_batches):
            X = test_features[i * batch_size: min(i * batch_size + batch_size, n_test_samples)]
            y = test_labels[i * batch_size: min(i * batch_size + batch_size, n_test_samples)]

            test_pred = classifier(X.to(device))

            test_logits = classifier(X.to(device))
            test_probs = torch.softmax(test_logits, dim=1)
            test_pred = test_probs.argmax(dim=1)

            test_loss += loss_fn(test_logits.to(device), y.to(device))
            test_acc += (torch.eq(y.to(device), test_pred.to(device)).sum().item() * 100 / len(y))

        test_loss /= n_test_batches
        test_acc /= n_test_batches

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    writer.add_scalar(f'Text Classifier/Loss/Train', train_loss, epoch)
    writer.add_scalar(f'Text Classifier/Loss/Test', test_loss, epoch)
    writer.add_scalar(f'Text Classifier/Accuracy/Test', test_acc, epoch)

torch.save(classifier.state_dict(), 'trained_models/3_1-classifier')
