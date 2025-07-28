import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    def __init__(self, data_path, target_col='class', test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
        }
        self.results = {}
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.trained_models = {}
        self._load_and_split_data()

    def _load_and_split_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[self.target_col])
        # Drop non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
            X = X.select_dtypes(include=[np.number])
        # Drop columns that are all-NaN
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"Dropping columns that are all-NaN: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
        # Impute missing values with median
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        # Diagnostics for NaNs
        n_nans = X.isna().sum().sum()
        print(f"Number of NaNs after imputation: {n_nans}")
        if n_nans > 0:
            print("Columns with NaNs after imputation:", X.columns[X.isna().any()].tolist())
            # Drop columns that are still all-NaN
            X = X.dropna(axis=1)
            print("Dropped columns with remaining NaNs.")
        y = df[self.target_col]
        # Drop any rows with remaining NaNs
        n_rows_before = X.shape[0]
        mask = X.isna().any(axis=1)
        if mask.sum() > 0:
            print(f"Dropping {mask.sum()} rows with remaining NaNs.")
            X = X[~mask]
            y = y[~mask]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

    def evaluate_models(self):
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            f1 = f1_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            auc_pr = average_precision_score(self.y_test, y_proba) if y_proba is not None else None
            self.results[name] = {
                'f1_score': f1,
                'confusion_matrix': cm,
                'auc_pr': auc_pr,
                'y_pred': y_pred,
                'y_proba': y_proba
            }

    def plot_precision_recall(self):
        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            y_proba = res['y_proba']
            if y_proba is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
                auc_pr = res['auc_pr']
                plt.plot(recall, precision, label=f'{name} (AUC-PR={auc_pr:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def get_results(self):
        # Return a summary dictionary for notebook display
        summary = {}
        for name, res in self.results.items():
            summary[name] = {
                'F1-Score': res['f1_score'],
                'AUC-PR': res['auc_pr'],
                'Confusion Matrix': res['confusion_matrix']
            }
        return summary

    def best_model(self):
        # Returns the name of the best model based on AUC-PR, then F1
        best = max(self.results.items(), key=lambda x: (x[1]['auc_pr'] if x[1]['auc_pr'] is not None else 0, x[1]['f1_score']))
        return best[0], self.results[best[0]]

    def save_best_model(self, output_path='../models/best_model.joblib'):
        """Save the best-performing model and its feature names to disk in the models directory."""
        os.makedirs('models', exist_ok=True)
        best_name, _ = self.best_model()
        best_model = self.trained_models[best_name]
        # Save model
        joblib.dump(best_model, output_path)
        # Save feature names
        feature_names = self.X_train.columns.tolist()
        joblib.dump(feature_names, '../models/model_features.joblib')
        print(f"Best model ('{best_name}') saved to {output_path}.")
        print(f"Feature names saved to models/model_features.joblib.")
