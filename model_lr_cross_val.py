from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import RepeatedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

from sklearn.linear_model import LinearRegression

dataset = pd.read_excel('train_final.xlsx')

from math import sqrt


rkfold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
kk = 0
evals = []
for i, (idx_train, idx_test) in enumerate(rkfold.split(dataset)):
    vectorizer_main = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5))
    vectorizers_answers = dict(
        [
            (nn, TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5)))
            for nn
            in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
        ]
    )
    train_df = dataset.iloc[idx_train]
    test_df = dataset.iloc[idx_test]

    X_train = pd.concat(
        [
            pd.DataFrame(vectorizer_main.fit_transform(train_df['ItemStem_Text'].apply(lambda x: str(x))).todense(), index=train_df.index, columns=vectorizer_main.vocabulary_)
        ] +
        [
            pd.DataFrame(vectorizers_answers[nn].fit_transform(train_df[f'Answer__{nn}'].apply(lambda x: str(x))).todense(), index=train_df.index, columns=[nn + k for k in vectorizers_answers[nn].vocabulary_])
            for nn
            in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
        ] +
        [
            pd.Series([len(str(v).split(' ')) for v in train_df['ItemStem_Text'].values], index=train_df['ItemStem_Text'].index, name='TextLength')
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            pd.DataFrame(vectorizer_main.transform(test_df['ItemStem_Text'].apply(lambda x: str(x))).todense(), index=test_df.index, columns=vectorizer_main.vocabulary_)
        ] +
        [
            pd.DataFrame(vectorizers_answers[nn].transform(test_df[f'Answer__{nn}'].apply(lambda x: str(x))).todense(), index=test_df.index, columns=[nn + k for k in vectorizers_answers[nn].vocabulary_])
            for nn
            in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
        ] +
        [
            pd.Series([len(str(v).split(' ')) for v in test_df['ItemStem_Text'].values], index=test_df['ItemStem_Text'].index, name='TextLength')
        ],
        axis=1
    )

    y_train_diff = dataset.iloc[idx_train]['Difficulty']
    y_train_time = dataset.iloc[idx_train]['Response_Time'] / 100
    y_test_diff = dataset.iloc[idx_test]['Difficulty']
    y_test_time = dataset.iloc[idx_test]['Response_Time'] / 100

    X_train.fillna(0.0, inplace=True)
    print(X_train)
    X_test.fillna(0.0, inplace=True)
    print(len(y_train_diff))

    model_diff = LinearRegression()
    model_time = LinearRegression()

    model_diff.fit(X_train, y_train_diff)
    model_time.fit(X_train, y_train_time)

    diff_e = y_test_diff
    time_e = y_test_time
    diff_r = model_diff.predict(X_test)
    time_r = model_diff.predict(X_test)

    res = {
        'pearson_diff': pearsonr(diff_e, diff_r).statistic,
        'pearson_time': pearsonr(time_e, time_r).statistic,
        'spearman_diff': spearmanr(diff_e, diff_r).statistic,
        'spearman_time': spearmanr(time_e, time_r).statistic,
        'rmse_diff': sqrt(mean_squared_error(diff_e, diff_r)),
        'rmse_time': sqrt(mean_squared_error(time_e, time_r)),
        'mae_diff': mean_absolute_error(diff_e, diff_r),
        'mae_time': mean_absolute_error(time_e, time_r),
    }
    evals.append(res)
    print(res)

final_out = {}
for key in evals[0]:
    final_out[key] = 0.0
for key in final_out:
    for elem in evals:
        final_out[key] += elem[key]
for key in final_out:
    final_out[key] = final_out[key] / 25

with open('5x5-baseline-lr.txt', 'w', encoding='utf8') as f:
    f.write(str(final_out))