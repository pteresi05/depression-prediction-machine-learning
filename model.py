import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import argparse

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import root_mean_squared_error, explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor  
from sklearn.base import BaseEstimator, TransformerMixin, clone


rand_st = 1 # for reproducibility
data_path = 'Tech_Use_Stress_Wellness.csv' # update if necessary

feature_select_dict = {
    0: "No Feature Selection",
    1: "SelectKBest with Mutual Information Regression",
    2: "INTERACT Feature Selection",
    3: "Recursive Feature Elimination (RFE)"
}

# Load and Process Data
ds = pd.read_csv(data_path)
target = "weekly_depression_score"
assert target in ds.columns, f"Variable of interest '{target}' not found."

# Drop ID and target variables that are not being used for prediction
ds.drop(['user_id', 'mental_health_score', 'weekly_anxiety_score'], axis=1, inplace=True)

boolean_cols = ['uses_wellness_apps', 'eats_healthy']
ds[boolean_cols] = ds[boolean_cols].astype(int)

ds["gender"] =  ds["gender"].map({"Male" : 1, "Female" : 0})
ds = pd.get_dummies(ds, columns=["location_type"], drop_first=True)

# Drop rows with missing values due to large dataset size
ds.dropna(inplace=True)


predictors=ds.drop(columns=ds["weekly_depression_score"].name)
target=ds["weekly_depression_score"]

np_predictors = np.array(predictors)
np_target = np.array(target)

# Split data into training and testing sets (65% train, 35% test) 
data_train, data_test, target_train, target_test = train_test_split(np_predictors, np_target, test_size=0.35, random_state=rand_st)


# Evaluation function to compute metrics on train and test sets
def evaluate(name, model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    return {
        "Model":    name,
        "RMSE":     round(root_mean_squared_error(y_test, test_pred), 4),
        "Expl Var": round(explained_variance_score(y_test, test_pred), 4),
    }

# cross-validate a model on training data and return mean CV scores
def cv_evaluate(model, X_train, y_train, cv=5):
    scorers={'Neg_MSE' : 'neg_root_mean_squared_error', 'expl_var': 'explained_variance'}
    kf = KFold(n_splits=cv, shuffle=True, random_state=rand_st)
    start_ts=time.time()
    cv_res = cross_validate(model, X_train, y_train, cv=kf, scoring=scorers)
    return {
        "CV Expl Var Mean": round(cv_res["test_expl_var"].mean(), 4),
        "CV RMSE Mean": round(-cv_res["test_Neg_MSE"].mean(), 4),
        "CV Runtime: ": time.time()-start_ts
    }

# ── SHAP Explainability ───────────────────────────────────────────────────────
def shap_explain(name, pipeline, X_test, feature_names, n_background=100):
    """Compute and plot SHAP values for a fitted sklearn pipeline."""
    # Get data after all steps except the final estimator
    X_transformed = pipeline[:-1].transform(X_test)
    estimator = pipeline[-1]

    # Tree-based models: fast exact explainer
    if isinstance(estimator, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Model-agnostic: use a small background sample for speed
        background = shap.sample(X_transformed, min(n_background, len(X_transformed)))
        explainer = shap.KernelExplainer(estimator.predict, background)
        shap_values = explainer.shap_values(X_transformed, nsamples=200)

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                      show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                      show=False)
    plt.title(f"SHAP Summary — {name}")
    plt.tight_layout()
    plt.show()


# calculates Shannon entropy of a discrete variable x
def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log2(probs + 1e-10))

# calculates mutual information between two discrete variables x and y
def mutual_info(x, y):
    n = len(x)
    xy = np.array(list(zip(x, y)))
    _, counts_xy = np.unique(xy, axis=0, return_counts=True)
    h_xy = -np.sum((counts_xy / n) * np.log2(counts_xy / n + 1e-10))
    return entropy(x) + entropy(y) - h_xy

# calculates symmetrical uncertainty between two discrete variables x and y
def symmetrical_uncertainty(x, y):
    mi = mutual_info(x, y)
    denom = entropy(x) + entropy(y)
    return 2 * mi / denom if denom > 0 else 0


class INTERACTSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, threshold=0.01):
        self.n_bins = n_bins
        self.threshold = threshold
        self.selected_idx_ = None

    # KBinsDiscretizer is used to convert continuous features into 
    # discrete bins for mutual information calculation
    def fit(self, X, y):
        disc = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        X_disc = disc.fit_transform(X).astype(int)
        y_disc = pd.cut(y, bins=self.n_bins, labels=False)
        self.selected_idx_ = interact(X_disc, y_disc, threshold=self.threshold)
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]


def interact(X, y, threshold=0.0):
    n_features = X.shape[1]
    su_class = [symmetrical_uncertainty(X[:, i], y) for i in range(n_features)]
    sorted_idx = np.argsort(su_class)[::-1]
    sorted_idx = [i for i in sorted_idx if su_class[i] > threshold]

    removed = set()
    selected = []
    for i, fi in enumerate(sorted_idx):
        if fi in removed:
            continue
        selected.append(fi)
        for fj in sorted_idx[i+1:]:
            if fj in removed:
                continue
            if symmetrical_uncertainty(X[:, fj], X[:, fi]) >= su_class[fj]:
                removed.add(fj)
    return selected

results = []

while True:
    try:
        fs_type = int(input("Specify feature selection method (0: None, 1: SelectKBest, 2: INTERACT, 3: RFE): "))
        if fs_type in feature_select_dict:
            break
        print(f"Invalid choice. Please enter a number between 0 and {len(feature_select_dict)-1}.")
    except ValueError:
        print("Invalid input. Please enter a number.")
print("=== Feature Selection Method:", feature_select_dict[fs_type], "===")

if fs_type == 0:

    rfrgr = RandomForestRegressor(n_estimators = 100, max_depth = None,
                        max_features = .33, min_samples_split=3, criterion='squared_error',
                        random_state = rand_st)

    rfrgr.fit(data_train, target_train)
    results.append({**evaluate("Random Forest", rfrgr,
                               data_train, target_train, data_test, target_test),
                    **cv_evaluate(rfrgr, data_train, target_train)})

    # 2. Gradient Boosting
    gbrgr = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=rand_st)
    gbrgr.fit(data_train, target_train)
    results.append({**evaluate("Gradient Boosting", gbrgr,
                               data_train, target_train, data_test, target_test),
                    **cv_evaluate(gbrgr, data_train, target_train)})

    # 3. Support Vector Regression
    #    SVR is sensitive to scale, so we wrap it in a pipeline with StandardScaler
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="linear", C=1.0, epsilon=0.1)),
    ])
    svr_pipe.fit(data_train, target_train)
    results.append({**evaluate("SVR (Linear)", svr_pipe,
                               data_train, target_train, data_test, target_test),
                    **cv_evaluate(svr_pipe, data_train, target_train)})
    
    # 4. MLP (Neural Network)
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                learning_rate_init=0.001,max_iter=500,
                                early_stopping=True,solver='adam',
                                n_iter_no_change=10,random_state=rand_st)),
    ])
    mlp_pipe.fit(data_train, target_train)
    results.append({**evaluate("MLP Regressor", mlp_pipe,
                               data_train, target_train, data_test, target_test),
                    **cv_evaluate(mlp_pipe, data_train, target_train)})

if fs_type == 1: # use KBest feature selection
    rgr = RandomForestRegressor(n_estimators = 100, max_depth = None, 
                                max_features = .33, min_samples_split=3, criterion='squared_error',
                                random_state = rand_st)      

    select = SelectKBest(score_func=mutual_info_regression, k=9)
    rgrpipe = Pipeline([
        ("select", select),
        ("rgr", rgr)])
    rgrpipe.fit(data_train, target_train)
    results.append({**evaluate("Random Forest (SelectKBest)", rgrpipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(rgrpipe, data_train, target_train)})
    
    grbreg = GradientBoostingRegressor(n_estimators=200, max_depth=None,
                                    learning_rate=0.05, random_state=rand_st)
    grbpipe = Pipeline([
        ("select", select),
        ("grb", grbreg)])
    grbpipe.fit(data_train, target_train)
    results.append({**evaluate("Gradient Boosting (SelectKBest)", grbpipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(grbpipe, data_train, target_train)})
    svr_pipe = Pipeline([
        ("select", select),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1))
    ]) 
    svr_pipe.fit(data_train, target_train)
    results.append({**evaluate("SVR (Linear, SelectKBest)", svr_pipe
                            , data_train, target_train, data_test, target_test),
                    **cv_evaluate(svr_pipe, data_train, target_train)})

    mlp_pipe=Pipeline([
        ("select", select),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                learning_rate_init=0.001,max_iter=500,
                                early_stopping=True,solver='adam',
                                n_iter_no_change=10,random_state=rand_st))
    ])
    mlp_pipe.fit(data_train, target_train)
    results.append({**evaluate("MLP Regressor (SelectKBest)", mlp_pipe
                            , data_train, target_train, data_test, target_test),
                    **cv_evaluate(mlp_pipe, data_train, target_train)})
    print(select.get_support())

    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_train[0])):
        if select.get_support()[i]:                                                           #Selected Features get added to temp header
            temp.append(predictors.columns[i])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected:', temp)
    print('Features (total/selected):', len(data_train[0]), len(temp))
    print('\n')

    print("\n=== SHAP Explainability (SelectKBest pipelines) ===")
    for name, pipe in [
        ("Random Forest (SelectKBest)", rgrpipe),
        ("Gradient Boosting (SelectKBest)", grbpipe),
        ("SVR (Linear, SelectKBest)", svr_pipe),
        ("MLP Regressor (SelectKBest)", mlp_pipe),
    ]:
        sel_idx = pipe.named_steps["select"].get_support(indices=True)
        feat_names = [predictors.columns[i] for i in sel_idx]
        print(f"  Running SHAP for {name}...")
        shap_explain(name, pipe, data_test, feat_names)
   
if fs_type == 2: # INTERACT Feature Selection 
    rf_pipe = Pipeline([
        ("interact", INTERACTSelector(n_bins=5, threshold=0.01)),
        ("rfrgr", RandomForestRegressor(n_estimators=100, max_depth=None,
                                        max_features=.33, min_samples_split=3,
                                        criterion='squared_error', random_state=rand_st))
    ])
    rf_pipe.fit(data_train, target_train)
    # Get selected feature indices from INTERACT and print their names
    interact_selector = rf_pipe.named_steps["interact"]
    results.append({**evaluate("Random Forest (INTERACT)", rf_pipe,
                                data_train, target_train, data_test, target_test),
                    **cv_evaluate(rf_pipe, data_train, target_train)})

    gb_pipe = Pipeline([
        ("interact", INTERACTSelector(n_bins=5, threshold=0.01)),
        ("gbrgr", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=rand_st))
    ])
    gb_pipe.fit(data_train, target_train)
    results.append({**evaluate("Gradient Boosting (INTERACT)", gb_pipe,
                                data_train, target_train, data_test, target_test),
                    **cv_evaluate(gb_pipe, data_train, target_train)})

    svr_pipe = Pipeline([
        ("interact", INTERACTSelector(n_bins=5, threshold=0.01)),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1))
    ])
    svr_pipe.fit(data_train, target_train)
    results.append({**evaluate("SVR (RBF, INTERACT)", svr_pipe,
                                data_train, target_train, data_test, target_test),
                    **cv_evaluate(svr_pipe, data_train, target_train)})

    mlp_pipe = Pipeline([
        ("interact", INTERACTSelector(n_bins=5, threshold=0.01)),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                max_iter=500, learning_rate_init=0.001,
                                early_stopping=True,solver='adam',
                                n_iter_no_change=10, random_state=rand_st))
    ])
    mlp_pipe.fit(data_train, target_train)
    results.append({**evaluate("MLP Regressor (INTERACT)", mlp_pipe,
                                data_train, target_train, data_test, target_test),
                    **cv_evaluate(mlp_pipe, data_train, target_train)})
    
    selected_indices = interact_selector.selected_idx_
    selected_features = [predictors.columns[i] for i in selected_indices]
    print("Selected features by INTERACT:", selected_features)

    print("\n=== SHAP Explainability (INTERACT pipelines) ===")
    for name, pipe in [
        ("Random Forest (INTERACT)",    rf_pipe),
        ("Gradient Boosting (INTERACT)", gb_pipe),
        ("SVR (RBF, INTERACT)",         svr_pipe),
        ("MLP Regressor (INTERACT)",    mlp_pipe),
    ]:
        sel_idx = pipe.named_steps["interact"].selected_idx_
        feat_names = [predictors.columns[i] for i in sel_idx]
        print(f"  Running SHAP for {name}...")
        shap_explain(name, pipe, data_test, feat_names)

if fs_type == 3:

    rf_estimator = RandomForestRegressor(n_estimators=100, max_depth=None,
                                        max_features=.33, min_samples_split=3,
                                        criterion='squared_error', random_state=rand_st)
    rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=None, step=1)

    rf_pipe = Pipeline([
        ("rfe", rfe_selector),
        ("rfrgr", RandomForestRegressor(n_estimators=100, max_depth=None,
                                        max_features=.33, min_samples_split=3,
                                        criterion='squared_error', random_state=rand_st))
    ])
    rf_pipe.fit(data_train, target_train)
    results.append({**evaluate("Random Forest (RFE)", rf_pipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(rf_pipe, data_train, target_train)})

    gb_rfe_estimator = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=rand_st)
    gbrgr = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=rand_st)

    gb_pipe = Pipeline([
        ("rfe", RFE(estimator=gb_rfe_estimator, n_features_to_select=10, step=1)),
        ("gbrgr", gbrgr)
    ])
    gb_pipe.fit(data_train, target_train)
    results.append({**evaluate("Gradient Boosting (RFE)", gb_pipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(gb_pipe, data_train, target_train)})

    svr_pipe = Pipeline([
        ("rfe", RFE(estimator=rf_estimator, n_features_to_select=10, step=1)),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1))
    ])
    svr_pipe.fit(data_train, target_train)
    results.append({**evaluate("SVR (Linear, RFE)", svr_pipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(svr_pipe, data_train, target_train)})
    
    mlp_pipe = Pipeline([
        ("rfe", RFE(estimator=rf_estimator, n_features_to_select=10, step=1)),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                learning_rate_init=0.001,max_iter=500,
                                early_stopping=True,solver='adam',
                                n_iter_no_change=10,random_state=rand_st))
    ])
    mlp_pipe.fit(data_train, target_train)
    results.append({**evaluate("MLP Regressor (RFE)", mlp_pipe,
                            data_train, target_train, data_test, target_test),
                    **cv_evaluate(mlp_pipe, data_train, target_train)})
    
    # create table of selected features for RFE
    rfe_selector = rf_pipe.named_steps["rfe"]
    selected_indices = rfe_selector.get_support(indices=True)
    selected_features = [predictors.columns[i] for i in selected_indices]
    print("Selected features by RFE:", selected_features)


    print("\n=== SHAP Explainability (RFE pipelines) ===")
    for name, pipe in [
        ("Random Forest (RFE)", rf_pipe),
        ("Gradient Boosting (RFE)", gb_pipe),
        ("SVR (Linear, RFE)", svr_pipe),
        ("MLP Regressor (RFE)", mlp_pipe),
    ]:
        sel_idx = pipe.named_steps["rfe"].get_support(indices=True)
        feat_names = [predictors.columns[i] for i in sel_idx]
        print(f"  Running SHAP for {name}...")
        shap_explain(name, pipe, data_test, feat_names)



# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).set_index("Model")
print("\n=== Model Comparison ===")
print(results_df.to_string())
