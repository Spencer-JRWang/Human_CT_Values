import warnings
warnings.filterwarnings('ignore')
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


def model_combinations():
    base_model = [
        ('RandomForest',RandomForestClassifier(n_estimators=2500)),
        #('GradientBoost',GradientBoostingClassifier(n_estimators=1000)),
        ('LGBM',LGBMClassifier(verbose = -1,n_estimators = 1000, max_depth = 5)),
        ('XGBoost',XGBClassifier(n_estimators = 1000, max_depth = 5)),
        ('CatBoost',CatBoostClassifier(verbose = False,iterations = 800, max_depth = 5))
    ]
    from itertools import combinations
    all_combinations = []
    for r in range(1, len(base_model) + 1):
        combinations_r = combinations(base_model, r)
        all_combinations.extend(combinations_r)
    return all_combinations

def stacking_model(X,y_encode,base_model):
    scores_st = []
    X = X.reset_index(drop=True)
    y_encode = y_encode.reset_index(drop=True)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    shuffle_index = np.random.permutation(X.index)
    X = X.iloc[shuffle_index]
    y_encode = y_encode.iloc[shuffle_index]
    meta_model = LogisticRegression(max_iter=10000000)
    stacking_clf = StackingClassifier(estimators=base_model, final_estimator=meta_model, stack_method='predict_proba')
    score_st = cross_val_predict(stacking_clf, X, y_encode, cv=stratified_kfold, method="predict_proba")
    scores_st.append(score_st[:, 1])
    scores_st = np.array(scores_st)
    scores_st = np.mean(scores_st, axis=0)
    dff = y_encode.to_frame()
    dff["IntegratedScore"] = scores_st
    return dff

def model_evaluation(model_list, X, y):
    for i in model_list:
        model_name = i[0]
        model = i[1]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y, cv=cv)
        print(f"Model {model_name} cross validation score is {score} with average {score.mean()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = i[1]
        model.fit(X_train, y_train)

        predicted_proba = model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicted_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"Model {model_name} cross train-test AUC is {roc_auc}")


