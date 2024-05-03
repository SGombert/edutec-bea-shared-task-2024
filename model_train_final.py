import torch

from transformers import AutoTokenizer, AutoModel
from allennlp.modules.scalar_mix import ScalarMix

from rational.torch import Rational

from tqdm import tqdm

import transformers

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

import numpy as np
import pandas as pd

from math import sqrt

device = torch.device('cuda')

def dataset_X(df, separation_token):
    outp = [''] * len(df)
    for row_index in range(len(df)):
        answer_str = f" {separation_token} ".join(
            [
                '(' + nn + ') ' + str(df[f"Answer__{nn}"].values[row_index]) + ('(Correct Answer)' if df[f"Answer_Key"].values[row_index] == nn else '')
                for nn
                in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
                if len(str(df[f"Answer__{nn}"].values[row_index])) > 0
            ]
        )

        outp[row_index] = f'{df["ItemType"].values[row_index]} {separation_token} {df["EXAM"].values[row_index]} {separation_token} {df["ItemStem_Text"].values[row_index]} {separation_token} {answer_str} {separation_token} Correct Solution: ({df[f"Answer_Key"].values[row_index]}) {df[f"Answer_Text"].values[row_index]}'
    return np.array(outp)

def dataset_y(df):
    return np.array(
        [
            np.array([df['Difficulty'].values[i], df['Response_Time'].values[i] / 100])
            for i
            in range(len(df))
        ]
    )

class ScoringModel(torch.nn.Module):
    def __init__(self, language_model, prefix) -> None:
        super().__init__()
        self.prefix = prefix

        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.lm = AutoModel.from_pretrained(language_model).to(device)
        self.scalar_mix_time = ScalarMix(self.lm.config.num_hidden_layers + 1)
        self.scalar_mix_difficulty = ScalarMix(self.lm.config.num_hidden_layers + 1)

        self.dropout = torch.nn.Dropout(p=0.2)

        self.lm_name = language_model

        self.regression_head_time = torch.nn.Sequential(
            torch.nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size),
            Rational(),
            torch.nn.Linear(self.lm.config.hidden_size, 1)
        )

        self.regression_head_difficulty = torch.nn.Sequential(
            torch.nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size),
            Rational(),
            torch.nn.Linear(self.lm.config.hidden_size, 1)
        )

        self.loss = torch.nn.MSELoss(reduction='mean')

        self.X = None
        self.y = None
        self.eval_X = None
        self.eval_y = None

    def set_dataset(self, X, y):
        self.X = X
        self.y = y

    def set_evalset(self, X, y):
        self.eval_X = X
        self.eval_y = y

    def self_eval(self):
        self.eval()

        ret = np.array([
            self.forward(elem).cpu().detach().numpy() for elem in tqdm(self.eval_X)
        ])
        diff_r = [r[0][0] for r in ret]
        diff_e = [float(x[0]) for x in self.eval_y]
        time_r = [r[0][1] for r in ret]
        time_e = [float(x[1]) for x in self.eval_y]

        return {
            'pearson_diff': pearsonr(diff_e, diff_r).statistic,
            'pearson_time': pearsonr(time_e, time_r).statistic,
            'spearman_diff': spearmanr(diff_e, diff_r).statistic,
            'spearman_time': spearmanr(time_e, time_r).statistic,
            'rmse_diff': sqrt(mean_squared_error(diff_e, diff_r)),
            'rmse_time': sqrt(mean_squared_error(time_e, time_r)),
            'mae_diff': mean_absolute_error(diff_e, diff_r),
            'mae_time': mean_absolute_error(time_e, time_r),
        }

    def forward(self, input):
        inputs = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=self.lm.config.max_position_embeddings - 2)
        outputs = self.lm(**inputs.to(device), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        result_difficulty = self.regression_head_difficulty(
            torch.mean(
                self.dropout(
                    self.scalar_mix_difficulty(
                        hidden_states
                    )
                ),
                dim=1
            )
        )
        
        result_time = self.regression_head_time(
            torch.mean(
                self.dropout(
                    self.scalar_mix_time(
                        hidden_states
                    )
                ),
                dim=1
            )
        )

        return torch.cat((result_difficulty, result_time), dim=1)

    def fit(
        self,
        epochs,
        optimizer,
        scheduler,
        batch_size=4
    ) -> None:
        self.train()
        for epoch in range(epochs):
            print(epoch)
            r = 0.0
            num_s = 0.0
            d, d_y = shuffle(self.X, self.y)

            batches_X = [
                d[n : n + batch_size] for n in range(0, len(d), batch_size)
            ]
            batches_y = [
                d_y[n:n + batch_size] for n in range(0, len(d_y), batch_size)
            ]

            for batch in tqdm(range(len(batches_X))):
                pred = self.forward(list(batches_X[batch]))
                ls = self.loss(pred, batches_y[batch])

                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
                scheduler.step()

                r += ls.detach().item()
                num_s += 1

                if batch % 10 == 0 and batch > 0:
                    print(str(r / num_s))

                if not (self.eval_X is None) and num_s % 50 == 0:
                    ev = self.self_eval()
                    print(ev)
                    self.train()

            if not (self.eval_X is None) and num_s % 50 == 0:
                ev = self.self_eval()
                print(ev)
                self.train()

dataset = pd.read_csv('train_final.csv')

X_tr = dataset_X(dataset, '</s> <s>')
Y_tr = dataset_y(dataset)

model = ScoringModel('FacebookAI/roberta-large', f'run').to(device)
model.set_dataset(X_tr, torch.tensor(Y_tr, dtype=torch.float).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=12 * len(X_tr) )

model.fit(3, optimizer, scheduler=scheduler, batch_size=1)
torch.save(model.state_dict(), 'final-model-roberta.pt')

X_tr = dataset_X(dataset, '[SEP]')
Y_tr = dataset_y(dataset)

model = ScoringModel('google/electra-large-discriminator', f'run').to(device)
model.set_dataset(X_tr, torch.tensor(Y_tr, dtype=torch.float).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=12 * len(X_tr) )

model.fit(3, optimizer, scheduler=scheduler, batch_size=1)
torch.save(model.state_dict(), 'final-model-electra.pt')

X_tr = dataset_X(dataset, '[SEP]')
Y_tr = dataset_y(dataset)

model = ScoringModel('microsoft/deberta-v3-large', f'run').to(device)
model.set_dataset(X_tr, torch.tensor(Y_tr, dtype=torch.float).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=12 * len(X_tr) )

model.fit(3, optimizer, scheduler=scheduler, batch_size=1)
torch.save(model.state_dict(), 'final-model-deberta.pt')