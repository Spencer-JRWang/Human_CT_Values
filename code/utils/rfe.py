import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


def model_rfe(f,core,df,cat_A,cat_B):
    X = df.drop("Disease",axis = 1)
    y = df['Disease']
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    shuffle_index = np.random.permutation(X.index)
    X = X.iloc[shuffle_index]
    y = y.iloc[shuffle_index]
    y_encode = y.map({cat_A: 0, cat_B: 1})
    outcome_feature = []
    outcome_score = []
    for i in tqdm(range(X.shape[1])):
        rfe = RFE(core, n_features_to_select=i + 1)
        rfe.fit(X, y_encode)
        selected_features = X.columns[rfe.support_]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(core, X[selected_features], y_encode, cv=cv)
        selected_features = X.columns[rfe.support_]
        outcome_feature.append(selected_features)
        outcome_score.append(scores.mean())
    max_predict_data = max(outcome_score)
    best_features = list(outcome_feature[outcome_score.index(max_predict_data)])
    f.write("Best Features Combination Detected: " + str(best_features) + "\n")
    f.write("Best Validation Score: " + str(max_predict_data) + "\n")
    print("Best Features Combination Detected: " + str(best_features))
    print("Best Validation Score: " + str(max_predict_data))

    return best_features

 
