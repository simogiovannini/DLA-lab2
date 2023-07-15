import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
from utils.ag_wrapper import AGWrapper


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset("ag_news")
train_dataset = AGWrapper(dataset['test'])

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2Model.from_pretrained('gpt2')

with torch.no_grad():
    X = torch.empty(0, 768)
    X.to(device)
    for i in range(len(train_dataset)):
        s, y = train_dataset[i]
        x = tokenizer(s, return_tensors='pt')
        x = gpt_model(**x)
        x = x.last_hidden_state
        x = torch.mean(x, 1)
        x.to(device)
        X = torch.cat((X, x), 0)
        del x
        torch.cuda.empty_cache()
        print(i + 1)

    print(X)
    print(X.shape)
    print(X.element_size() * X.nelement())

    torch.save(X, 'datasets/ag_news_test_post_gpt')
