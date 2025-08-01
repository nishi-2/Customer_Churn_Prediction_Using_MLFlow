import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix



df = pd.read_csv("data/churn.csv")


# Dropping Customer ID 
df = df.drop('customerID', axis=1)


# TotalCharges is in number but coming as object type, changing it to number
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].isna().sum() # Null value is observed after coercing because empty strings were there that converted to NaN
mean_total_charge = df['TotalCharges'].mean()
df['TotalCharges'] = df['TotalCharges'].fillna(mean_total_charge) # Filling NaN with mean



# Label Encoding Target variable
target_le = LabelEncoder()
Y = target_le.fit_transform(df['Churn'])


# Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], Y, test_size=0.2, stratify=Y, random_state=42)


# Scaling Numerical features and encoding categorical features - OHE
categorical_cols = df.iloc[:, :-1].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.iloc[:, :-1].select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor_OHE = ColumnTransformer(
    transformers= [
        ('numerical', StandardScaler(), numerical_cols),
        ('categorical', OneHotEncoder(), categorical_cols)
    ]
)
X_train_processed = preprocessor_OHE.fit_transform(x_train)
X_test_processed = preprocessor_OHE.transform(x_test)



# Scaling Numerical features and encoding categorical features - LabelEncode
x_train_encoded = x_train.copy()
x_test_encoded = x_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    x_train_encoded[col] = le.fit_transform(x_train[col])
    x_test_encoded[col] = le.transform(x_test[col])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_cols)
    ],
    remainder='passthrough'  # Keep label encoded categorical columns as it is
)

X_train_processed_labelenc = preprocessor.fit_transform(x_train_encoded)
X_test_processed_labelenc = preprocessor.transform(x_test_encoded)



# Feature selection - KBest
selector_kbest = SelectKBest(score_func=mutual_info_classif, k=12)
X_train_kbest = selector_kbest.fit_transform(X_train_processed, y_train)
X_test_kbest = selector_kbest.transform(X_test_processed)


# Feature selection - RandomForest
selector_model = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
X_train_rf = selector_model.fit_transform(X_train_processed, y_train)
X_test_rf = selector_model.transform(X_test_processed)


# Feature selection - PCA
pca = PCA(n_components=0.90, random_state=42)  # retain 90% variance
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)



# Resampling Training Data
def return_oversample_undersample_data(x_train_data, y_train_data):
    train_df = pd.concat([pd.DataFrame(x_train_data), pd.Series(y_train_data, name='target')], axis=1)

    # Separate majority and minority classes
    majority = train_df[train_df.target == train_df.target.value_counts().idxmax()]
    minority = train_df[train_df.target == train_df.target.value_counts().idxmin()]

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42) # Upsample minority class
    oversampled_df = pd.concat([majority, minority_upsampled]) # Combine back

    # Separate features and labels
    X_train_oversampled = oversampled_df.drop('target', axis=1).values
    y_train_oversampled = oversampled_df['target'].values


    majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    undersampled_df = pd.concat([majority_downsampled, minority])
    X_train_undersampled = undersampled_df.drop('target', axis=1).values
    y_train_undersampled = undersampled_df['target'].values

    return X_train_oversampled, y_train_oversampled, X_train_undersampled, y_train_undersampled



X_train_oversampled_kbest, y_train_oversampled_kbest, X_train_undersampled_kbest, y_train_undersampled_kbest = return_oversample_undersample_data(X_train_kbest, y_train)
X_train_oversampled_rf, y_train_oversampled_rf, X_train_undersampled_rf, y_train_undersampled_rf = return_oversample_undersample_data(X_train_rf, y_train)
X_train_oversampled_pca, y_train_oversampled_pca, X_train_undersampled_pca, y_train_undersampled_pca = return_oversample_undersample_data(X_train_pca, y_train)



# Model training and evaluation
param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}


def train_and_evaluate_model(param_grids, x_train_sample, y_train_sample, x_test_sample, y_test_sample):
    results_dict = {}
    for name, config in param_grids.items():
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1)
        grid.fit(x_train_sample, y_train_sample)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test_sample)
        y_prob = best_model.predict_proba(x_test_sample)[:, 1] if hasattr(best_model, "predict_proba") else None

        results_dict[name] = {
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test_sample, y_pred),
        "precision": precision_score(y_test_sample, y_pred),
        "recall": recall_score(y_test_sample, y_pred),
        "f1_score": f1_score(y_test_sample, y_pred),
        "roc_auc": roc_auc_score(y_test_sample, y_prob) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test_sample, y_pred),
        "model": best_model
        }

    #for model_name, metrics in results_dict.items():
    #    print(f"\nModel: {model_name}")
    #    for k, v in metrics.items():
    #        if k != "model":
    #            print(f"{k}: {v}")

    return results_dict
        

results_dict_kbest_oversample = train_and_evaluate_model(param_grids, X_train_oversampled_kbest, y_train_oversampled_kbest, X_test_kbest, y_test)
results_dict_kbest_undersample = train_and_evaluate_model(param_grids, X_train_undersampled_kbest, y_train_undersampled_kbest, X_test_kbest, y_test)
results_dict_pca_oversample = train_and_evaluate_model(param_grids, X_train_oversampled_pca, y_train_oversampled_pca, X_test_pca, y_test)
results_dict_pca_undersample = train_and_evaluate_model(param_grids, X_train_undersampled_pca, y_train_undersampled_pca, X_test_pca, y_test)
results_dict_rf_oversample = train_and_evaluate_model(param_grids, X_train_oversampled_rf, y_train_oversampled_rf, X_test_rf, y_test)
results_dict_rf_undersample = train_and_evaluate_model(param_grids, X_train_undersampled_rf, y_train_undersampled_rf, X_test_rf, y_test)
results_dict_no_sampling = train_and_evaluate_model(param_grids, X_train_processed, y_train, X_test_processed, y_test)
results_dict_no_sampling_labelenc = train_and_evaluate_model(param_grids, X_train_processed_labelenc, y_train, X_test_processed_labelenc, y_test)


# Using MLFlow to Track
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("Customer_Churn_Prediction")

all_results = {
    "rf_oversample": results_dict_rf_oversample,
    "rf_undersample": results_dict_rf_undersample,
    "pca_oversample": results_dict_pca_oversample,
    "pca_undersample": results_dict_pca_undersample,
    "kbest_oversample": results_dict_kbest_oversample,
    "kbest_undersample": results_dict_kbest_undersample,
    "no_sampling_OHE" : results_dict_no_sampling,
    "no_sampling_LabelEncode" : results_dict_no_sampling_labelenc
}

best_model = None
best_score = -1
best_model_name = ""
best_run_id = ""

for variant_name, model_dict in all_results.items():
    for model_name, metrics in model_dict.items():
        with mlflow.start_run(run_name=f"{variant_name}_{model_name}"):
            # Log parameters
            for param_name, param_value in metrics["best_params"].items():
                mlflow.log_param(param_name, param_value)

            # Log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            if metrics["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", metrics["roc_auc"])

            # Log model
            mlflow.sklearn.log_model(metrics["model"], name="model", registered_model_name=None)

            # Update best model based on precision_score
            if metrics["precision"] > best_score:
                best_score = metrics["precision"]
                best_model = metrics["model"]
                best_model_name = f"{variant_name}_{model_name}"
                best_run_id = mlflow.active_run().info.run_id

        mlflow.end_run()


# Registering the best model in MLFlow Register
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="Best_Model_Classifier"
)

print(f"Best model registered: {best_model_name} with precision score: {best_score:.4f}")


# Saving the best model locally
import joblib
joblib.dump(best_model, "best_model.pkl")


# Saving preprocessors

# Save OneHotEncoder-based preprocessor
joblib.dump(preprocessor_OHE, 'preprocessor_ohe.pkl')

# Save LabelEncoder-based transformation dict for each column
categorical_cols = df.iloc[:, :-1].select_dtypes(include=['object', 'category']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(x_train[col])
    label_encoders[col] = le
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save final ColumnTransformer using label-encoded data
joblib.dump(preprocessor, 'preprocessor_label_encoded.pkl')
