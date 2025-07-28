import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ModelExplainability:
    def __init__(self, model_path: str, feature_names_path: str, data_path: str):
        """
        Initialize ModelExplainability for fraud detection analysis.
        
        Args:
            model_path: Path to the trained model (.joblib file)
            feature_names_path: Path to the feature names file (.joblib file)
            data_path: Path to the processed data file (.csv file)
        """
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.data = None
        self.X = None
        self.y = None
        self.explainer = None
        self.shap_values = None
        self.explanation = None

    def load_model_and_data(self, sample_size: Optional[int] = None, target_column: str = 'is_fraud'):
        """
        Load model, feature names, and data. Optionally sample the data for speed.
        
        Args:
            sample_size: Number of samples to use for SHAP analysis (None for all data)
            target_column: Name of the target column in the dataset
        """
        try:
            # Load the trained model
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            
            # Load feature names
            self.feature_names = joblib.load(self.feature_names_path)
            print(f"✓ Feature names loaded successfully from {self.feature_names_path}")
            
            # Load the data
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully from {self.data_path}")
            print(f"  - Dataset shape: {self.data.shape}")
            
            # Extract features and target
            if target_column in self.data.columns:
                self.y = self.data[target_column]
                self.X = self.data[self.feature_names]
                print(f"✓ Target column '{target_column}' found")
                print(f"  - Fraud cases: {self.y.sum()} ({self.y.mean()*100:.2f}%)")
                print(f"  - Non-fraud cases: {(self.y == 0).sum()} ({(self.y == 0).mean()*100:.2f}%)")
            else:
                self.X = self.data[self.feature_names]
                print(f"⚠ Target column '{target_column}' not found, using all data for SHAP analysis")
            
            # Sample data if requested
            if sample_size is not None and sample_size < len(self.X):
                # Ensure we have a good mix of fraud and non-fraud cases
                if self.y is not None:
                    fraud_indices = self.y[self.y == 1].index
                    non_fraud_indices = self.y[self.y == 0].index
                    
                    # Sample fraud cases (up to 50% of sample_size)
                    fraud_sample_size = min(len(fraud_indices), sample_size // 2)
                    non_fraud_sample_size = sample_size - fraud_sample_size
                    
                    fraud_sample = fraud_indices[:fraud_sample_size]
                    non_fraud_sample = non_fraud_indices[:non_fraud_sample_size]
                    
                    sample_indices = fraud_sample.union(non_fraud_sample)
                    self.X = self.X.loc[sample_indices]
                    if self.y is not None:
                        self.y = self.y.loc[sample_indices]
                else:
                    self.X = self.X.sample(n=sample_size, random_state=42)
                
                print(f"✓ Sampled {len(self.X)} instances for SHAP analysis")
            
            print(f"✓ Final feature matrix shape: {self.X.shape}")
            
        except Exception as e:
            print(f"❌ Error loading model and data: {str(e)}")
            raise

    def compute_shap_values(self, background_size: int = 100):
        """
        Compute SHAP values using appropriate explainer for the model type.
        
        Args:
            background_size: Number of background samples for KernelExplainer
        """
        if self.model is None or self.X is None:
            raise ValueError("Model and data must be loaded first.")
        
        try:
            # Determine the best explainer based on model type
            model_type = type(self.model).__name__
            print(f"✓ Computing SHAP values for {model_type} model...")
            
            if hasattr(self.model, 'feature_importances_') or 'tree' in model_type.lower():
                # Tree-based models (Random Forest, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X)
                print("✓ Using TreeExplainer")
            else:
                # Other models (Logistic Regression, SVM, etc.)
                # Use a background dataset for KernelExplainer
                background_data = self.X.sample(n=min(background_size, len(self.X)), random_state=42)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
                self.shap_values = self.explainer.shap_values(self.X)
                print("✓ Using KernelExplainer")
            
            print("✓ SHAP values computed successfully")
            
        except Exception as e:
            print(f"❌ Error computing SHAP values: {str(e)}")
            raise

    def get_correct_shap_values(self):
        """Return the correct SHAP values array for plotting."""
        if isinstance(self.shap_values, list):
            # Binary classification: use class 1 (fraud)
            return self.shap_values[1]
        return self.shap_values

    def plot_summary(self, max_display: int = 20, save_path: Optional[str] = None):
        """
        Create a summary plot showing global feature importance.
        
        Args:
            max_display: Maximum number of features to display
            save_path: Path to save the plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        shap_values = self.get_correct_shap_values()
        
        plt.figure(figsize=(12, 8))
        
        # Try different SHAP APIs
        try:
            # Try the legacy summary_plot first (most reliable)
            shap.summary_plot(shap_values, self.X, feature_names=self.feature_names, 
                             max_display=max_display, show=False)
        except Exception as e:
            print(f"Warning: Could not create summary plot with SHAP API: {e}")
            # Fallback to custom summary plot
            self._create_custom_summary_plot(shap_values, max_display)
        
        plt.title("SHAP Summary Plot - Global Feature Importance", fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved to {save_path}")
        
        plt.show()
        return plt.gcf()

    def _create_custom_summary_plot(self, shap_values, max_display):
        """Create a custom summary plot when SHAP API fails."""
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top features - ensure indices are integers
        top_indices = np.argsort(mean_abs_shap)[-max_display:].astype(int)
        top_features = [self.feature_names[int(i)] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_values)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance (Mean Absolute SHAP Values)')

    def _create_custom_waterfall_plot(self, shap_values_instance, base_value, index):
        """Create a custom waterfall plot when SHAP API fails."""
        # Sort features by absolute SHAP value
        feature_contributions = list(zip(self.feature_names, shap_values_instance))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 15 features
        top_features = feature_contributions[:15]
        feature_names = [f[0] for f in top_features]
        contributions = [f[1] for f in top_features]
        
        # Calculate cumulative values
        cumulative = [base_value]
        for contrib in contributions:
            cumulative.append(cumulative[-1] + contrib)
        
        # Create waterfall plot
        plt.bar(range(len(feature_names)), contributions, 
               color=['red' if c > 0 else 'blue' for c in contributions], alpha=0.7)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.ylabel('SHAP Value')
        plt.title(f'Waterfall Plot - Instance {index}\n(Base: {base_value:.4f}, Final: {cumulative[-1]:.4f})')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, contrib in enumerate(contributions):
            plt.text(i, contrib + (0.01 if contrib > 0 else -0.01), f'{contrib:.4f}', 
                    ha='center', va='bottom' if contrib > 0 else 'top', fontsize=8)

    def _create_custom_dependence_plot(self, shap_values, feature_name, feature_index):
        """Create a custom dependence plot when SHAP API fails."""
        # Get feature values and corresponding SHAP values
        feature_values = self.X.iloc[:, feature_index]
        shap_values_for_feature = shap_values[:, feature_index]
        
        # Create scatter plot
        plt.scatter(feature_values, shap_values_for_feature, alpha=0.6, s=20)
        plt.xlabel(feature_name)
        plt.ylabel('SHAP Value')
        plt.title(f'Dependence Plot - {feature_name}')
        plt.grid(True, alpha=0.3)

    def _create_custom_beeswarm_plot(self, shap_values, max_display):
        """Create a custom beeswarm plot when SHAP API fails."""
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-max_display:]
        top_features = [self.feature_names[i] for i in top_indices]
        
        # Create a simplified beeswarm-like plot
        plt.figure(figsize=(10, 8))
        
        for i, feature_idx in enumerate(top_indices):
            feature_shap_values = shap_values[:, feature_idx]
            # Add some jitter for better visualization
            jitter = np.random.normal(0, 0.1, len(feature_shap_values))
            plt.scatter(feature_shap_values + jitter, [i] * len(feature_shap_values), 
                       alpha=0.6, s=10, c=feature_shap_values, cmap='RdBu')
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('SHAP Value')
        plt.title('Feature Impact Distribution (Beeswarm-like)')
        plt.grid(axis='x', alpha=0.3)

    def plot_force(self, index: int = 0, save_path: Optional[str] = None):
        """
        Create a force plot for a single instance showing local feature contributions.
        
        Args:
            index: Index of the instance to explain
            save_path: Path to save the plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        if index >= len(self.X):
            raise ValueError(f"Index {index} out of range. Dataset has {len(self.X)} instances.")
        
        shap_values = self.get_correct_shap_values()
        
        # Get the correct base value for the model
        if isinstance(self.explainer.expected_value, (list, tuple)):
            base_value = self.explainer.expected_value[1]  # For binary classification, use class 1
        else:
            base_value = self.explainer.expected_value
        
        plt.figure(figsize=(12, 6))
        
        # For now, let's create a simple bar plot showing feature contributions
        # This is a workaround for the force plot issues
        feature_contributions = list(zip(self.feature_names, shap_values[index]))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 10 features
        top_features = feature_contributions[:10]
        feature_names = [f[0] for f in top_features]
        contributions = [f[1] for f in top_features]
        
        # Create color coding: red for positive, blue for negative
        colors = ['red' if c > 0 else 'blue' for c in contributions]
        
        plt.barh(range(len(feature_names)), contributions, color=colors, alpha=0.7)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('SHAP Value (Feature Contribution)')
        plt.title(f'SHAP Feature Contributions - Instance {index}\n(Base Value: {base_value:.4f})')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, contrib) in enumerate(zip(feature_names, contributions)):
            plt.text(contrib + (0.01 if contrib > 0 else -0.01), i, f'{contrib:.4f}', 
                    va='center', ha='left' if contrib > 0 else 'right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Force plot (bar chart) saved to {save_path}")
        
        plt.show()
        return plt.gcf()

    def plot_waterfall(self, index: int = 0, save_path: Optional[str] = None):
        """
        Create a waterfall plot for a single instance.
        
        Args:
            index: Index of the instance to explain
            save_path: Path to save the plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        if index >= len(self.X):
            raise ValueError(f"Index {index} out of range. Dataset has {len(self.X)} instances.")
        
        shap_values = self.get_correct_shap_values()
        
        # Get the correct base value for the model
        if isinstance(self.explainer.expected_value, (list, tuple)):
            base_value = self.explainer.expected_value[1]  # For binary classification, use class 1
        else:
            base_value = self.explainer.expected_value
        
        plt.figure(figsize=(12, 8))
        
        # Try different SHAP APIs
        try:
            # Try the legacy waterfall_plot first
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[index],
                base_values=base_value,
                data=self.X.iloc[index].values,
                feature_names=self.feature_names
            ), show=False)
        except Exception as e:
            print(f"Warning: Could not create waterfall plot with SHAP API: {e}")
            # Fallback to custom waterfall plot
            self._create_custom_waterfall_plot(shap_values[index], base_value, index)
        
        plt.title(f"SHAP Waterfall Plot - Instance {index}", fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Waterfall plot saved to {save_path}")
        
        plt.show()
        return plt.gcf()

    def plot_dependence(self, feature_name: str, interaction_index: Optional[str] = None, 
                       save_path: Optional[str] = None):
        """
        Create a dependence plot for a specific feature.
        
        Args:
            feature_name: Name of the feature to analyze
            interaction_index: Feature to show interaction with (optional)
            save_path: Path to save the plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")
        
        shap_values = self.get_correct_shap_values()
        feature_index = self.feature_names.index(feature_name)
        
        plt.figure(figsize=(10, 6))
        
        # Try different SHAP APIs
        try:
            # Try the legacy dependence_plot first
            shap.dependence_plot(feature_index, shap_values, self.X, 
                               feature_names=self.feature_names, show=False)
        except Exception as e:
            print(f"Warning: Could not create dependence plot with SHAP API: {e}")
            # Fallback to custom dependence plot
            self._create_custom_dependence_plot(shap_values, feature_name, feature_index)
        
        plt.title(f"SHAP Dependence Plot - {feature_name}", fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dependence plot saved to {save_path}")
        
        plt.show()
        return plt.gcf()

    def plot_beeswarm(self, max_display: int = 20, save_path: Optional[str] = None):
        """
        Create a beeswarm plot showing the distribution of SHAP values.
        
        Args:
            max_display: Maximum number of features to display
            save_path: Path to save the plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        shap_values = self.get_correct_shap_values()
        
        plt.figure(figsize=(12, 8))
        
        # Try different SHAP APIs
        try:
            # Try the legacy summary_plot with plot_type="dot" for beeswarm effect
            shap.summary_plot(shap_values, self.X, feature_names=self.feature_names, 
                             plot_type="dot", max_display=max_display, show=False)
        except Exception as e:
            print(f"Warning: Could not create beeswarm plot with SHAP API: {e}")
            # Fallback to custom beeswarm plot
            self._create_custom_beeswarm_plot(shap_values, max_display)
        
        plt.title("SHAP Beeswarm Plot - Feature Impact Distribution", fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Beeswarm plot saved to {save_path}")
        
        plt.show()
        return plt.gcf()

    def get_feature_importance_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get a ranking of features by their mean absolute SHAP values.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        shap_values = self.get_correct_shap_values()
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create ranking DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).head(top_n)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df = importance_df[['rank', 'feature', 'mean_abs_shap']]
        
        return importance_df

    def analyze_fraud_instances(self, n_instances: int = 5) -> pd.DataFrame:
        """
        Analyze the top fraud instances and their key contributing features.
        
        Args:
            n_instances: Number of fraud instances to analyze
            
        Returns:
            DataFrame with fraud instance analysis
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Run compute_shap_values() first.")
        
        if self.y is None:
            raise ValueError("Target variable not available. Cannot identify fraud instances.")
        
        # Get fraud instances
        fraud_indices = self.y[self.y == 1].index
        if len(fraud_indices) == 0:
            print("⚠ No fraud instances found in the dataset.")
            return pd.DataFrame()
        
        # Get top fraud instances by SHAP values (highest probability of fraud)
        shap_values = self.get_correct_shap_values()
        fraud_shap_values = shap_values[fraud_indices]
        base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, tuple)) else self.explainer.expected_value
        fraud_probabilities = fraud_shap_values.sum(axis=1) + base_value
        
        # Get top instances
        top_fraud_indices = fraud_indices[np.argsort(fraud_probabilities)[-n_instances:]]
        
        results = []
        for i, idx in enumerate(top_fraud_indices):
            instance_shap = shap_values[idx]
            instance_data = self.X.iloc[idx]
            
            # Get top contributing features
            feature_contributions = list(zip(self.feature_names, instance_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 5 positive and negative contributors
            positive_contributors = [f"{feat}: {val:.4f}" for feat, val in feature_contributions if val > 0][:5]
            negative_contributors = [f"{feat}: {val:.4f}" for feat, val in feature_contributions if val < 0][:5]
            
            results.append({
                'instance_id': idx,
                'rank': i + 1,
                'fraud_probability': fraud_probabilities[idx],
                'top_positive_features': '; '.join(positive_contributors),
                'top_negative_features': '; '.join(negative_contributors),
                'total_contribution': fraud_probabilities[idx] - base_value
            })
        
        return pd.DataFrame(results)

    def generate_comprehensive_report(self, output_dir: str = "shap_analysis"):
        """
        Generate a comprehensive SHAP analysis report with all plots and insights.
        
        Args:
            output_dir: Directory to save the analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔍 Generating comprehensive SHAP analysis report...")
        
        # Generate all plots
        self.plot_summary(save_path=f"{output_dir}/summary_plot.png")
        self.plot_beeswarm(save_path=f"{output_dir}/beeswarm_plot.png")
        
        # Generate force plots for a few instances
        for i in range(min(3, len(self.X))):
            self.plot_force(i, save_path=f"{output_dir}/force_plot_instance_{i}.png")
            self.plot_waterfall(i, save_path=f"{output_dir}/waterfall_plot_instance_{i}.png")
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking()
        importance_df.to_csv(f"{output_dir}/feature_importance_ranking.csv", index=False)
        
        # Analyze fraud instances
        fraud_analysis = self.analyze_fraud_instances()
        if not fraud_analysis.empty:
            fraud_analysis.to_csv(f"{output_dir}/fraud_instances_analysis.csv", index=False)
        
        # Generate dependence plots for top features
        top_features = importance_df['feature'].head(5).tolist()
        for feature in top_features:
            try:
                self.plot_dependence(feature, save_path=f"{output_dir}/dependence_plot_{feature}.png")
            except Exception as e:
                print(f"⚠ Could not generate dependence plot for {feature}: {e}")
        
        print(f"✓ Comprehensive report generated in '{output_dir}' directory")
        
        return {
            'importance_ranking': importance_df,
            'fraud_analysis': fraud_analysis,
            'output_directory': output_dir
        }
