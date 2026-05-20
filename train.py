import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    df['over'] = df['over'].replace(0, 0.1)
    df['crr'] = df['current_score'] / (df['over'] + df['ball'] / 6)
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])

    pipe.fit(X, y)
    return pipe


if __name__ == "__main__":
    print("Training model...")
    pipe = train_model()
    joblib.dump(pipe, "pipe.pkl")
    print("Done! Model saved to pipe.pkl")