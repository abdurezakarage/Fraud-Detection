
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighCardinalityEncoder:
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        
    def fit(self, X, y=None):
        """Fit the encoder on the data."""
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                # For non-object columns, just pass through
                pass
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Transform the data."""
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X_transformed.values
    
    def get_feature_names_out(self):
        """Get feature names."""
        return np.array(self.feature_names)

class DataTransformer:
    
    def __init__(self, data_path: str, output_dir: str = "data/transformed"):

        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        
        # Preprocessing objects
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.preprocessor = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self) -> None:
  
        try:
            logger.info(f"Loading data from {self.data_path}...")
            self.data = pd.read_csv(self.data_path)
            
            # Convert datetime columns
            datetime_columns = ['signup_time', 'purchase_time', 'first_transaction', 'last_transaction']
            for col in datetime_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])
            
            logger.info(f"Data loaded successfully: {self.data.shape}")
            logger.info(f"Class distribution: {self.data['class'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_class_distribution(self) -> Dict:
     
        logger.info("Analyzing class distribution...")
        
        class_counts = self.data['class'].value_counts()
        class_percentages = self.data['class'].value_counts(normalize=True) * 100
        
        analysis = {
            'total_samples': len(self.data),
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'minority_samples': class_counts.min(),
            'majority_samples': class_counts.max()
        }
        
        logger.info(f"Class distribution analysis:")
        logger.info(f"  Total samples: {analysis['total_samples']}")
        logger.info(f"  Class counts: {analysis['class_counts']}")
        logger.info(f"  Class percentages: {analysis['class_percentages']}")
        logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}:1")
        
        return analysis
    
    def plot_class_distribution(self, save_plot: bool = True) -> None:
        """Plot the class distribution."""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Bar plot
        plt.subplot(1, 2, 1)
        class_counts = self.data['class'].value_counts()
        colors = ['#2E8B57', '#DC143C']  # Green for non-fraud, Red for fraud
        bars = plt.bar(class_counts.index, class_counts.values, color=colors)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Number of Samples')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Pie chart
        plt.subplot(1, 2, 2)
        class_percentages = self.data['class'].value_counts(normalize=True) * 100
        labels = ['Non-Fraud (0)', 'Fraud (1)']
        plt.pie(class_percentages.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'class_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {plot_path}")
        
        plt.show()
    
    def identify_feature_types(self) -> Dict[str, List[str]]:
      
        logger.info("Identifying feature types...")
        
        # Drop target and ID columns
        exclude_columns = ['class', 'user_id', 'signup_time', 'purchase_time', 
                          'first_transaction', 'last_transaction']
        
        feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        # Categorical features (object type or low cardinality)
        categorical_features = []
        numerical_features = []
        high_cardinality_features = []
        
        for col in feature_columns:
            if self.data[col].dtype == 'object':
                # Check cardinality for object columns
                unique_count = self.data[col].nunique()
                if unique_count > 100:  # High cardinality - use label encoding or drop
                    high_cardinality_features.append(col)
                    logger.warning(f"High cardinality feature '{col}' with {unique_count} unique values - will be label encoded")
                else:
                    categorical_features.append(col)
            elif self.data[col].nunique() <= 10 and col not in ['age', 'purchase_value']:
                # Low cardinality numerical features that should be treated as categorical
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        feature_types = {
            'categorical': categorical_features,
            'numerical': numerical_features,
            'high_cardinality': high_cardinality_features,
            'datetime': ['signup_time', 'purchase_time', 'first_transaction', 'last_transaction'],
            'target': ['class'],
            'id': ['user_id']
        }
        
        logger.info(f"Feature types identified:")
        logger.info(f"  Categorical features: {len(feature_types['categorical'])}")
        logger.info(f"  Numerical features: {len(feature_types['numerical'])}")
        logger.info(f"  High cardinality features: {len(feature_types['high_cardinality'])}")
        logger.info(f"  Datetime features: {len(feature_types['datetime'])}")
        
        return feature_types
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
       
        logger.info("Splitting data into training and testing sets...")
        
        # Identify feature columns
        feature_types = self.identify_feature_types()
        feature_columns = feature_types['categorical'] + feature_types['numerical'] + feature_types['high_cardinality']
        
        X = self.data[feature_columns].copy()
        y = self.data['class'].copy()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Testing set: {self.X_test.shape}")
        logger.info(f"  Training class distribution: {self.y_train.value_counts().to_dict()}")
        logger.info(f"  Testing class distribution: {self.y_test.value_counts().to_dict()}")
    
    def handle_class_imbalance(self, method: str = 'smote', random_state: int = 42) -> None:
      
        logger.info(f"Handling class imbalance using {method.upper()}...")
        
        # Get original class distribution
        original_distribution = self.y_train.value_counts()
        logger.info(f"Original training class distribution: {original_distribution.to_dict()}")
        
        # Apply sampling technique
        if method.lower() == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif method.lower() == 'adasyn':
            sampler = ADASYN(random_state=random_state, n_neighbors=5)
        elif method.lower() == 'random_under':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method.lower() == 'smoteenn':
            sampler = SMOTEENN(random_state=random_state)
        elif method.lower() == 'smotetomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Apply sampling
        self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(
            self.X_train, self.y_train
        )
        
        # Get new class distribution
        balanced_distribution = self.y_train_balanced.value_counts()
        logger.info(f"Balanced training class distribution: {balanced_distribution.to_dict()}")
        
        # Calculate improvement
        original_ratio = original_distribution.max() / original_distribution.min()
        balanced_ratio = balanced_distribution.max() / balanced_distribution.min()
        
        logger.info(f"Imbalance ratio improvement: {original_ratio:.2f}:1 → {balanced_ratio:.2f}:1")
        
        # Justify the choice
        self._justify_sampling_choice(method, original_ratio, balanced_ratio)
    
    def handle_class_imbalance_encoded(self, X_train_encoded: np.ndarray, y_train: pd.Series, 
                                     method: str = 'smote', random_state: int = 42) -> None:
       
        logger.info(f"Handling class imbalance on encoded data using {method.upper()}...")
        
        # Get original class distribution
        original_distribution = y_train.value_counts()
        logger.info(f"Original training class distribution: {original_distribution.to_dict()}")
        
        # Apply sampling technique
        if method.lower() == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif method.lower() == 'adasyn':
            sampler = ADASYN(random_state=random_state, n_neighbors=5)
        elif method.lower() == 'random_under':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method.lower() == 'smoteenn':
            sampler = SMOTEENN(random_state=random_state)
        elif method.lower() == 'smotetomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Apply sampling on encoded data
        self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(
            X_train_encoded, y_train
        )
        
        # Get new class distribution
        balanced_distribution = pd.Series(self.y_train_balanced).value_counts()
        logger.info(f"Balanced training class distribution: {balanced_distribution.to_dict()}")
        
        # Calculate improvement
        original_ratio = original_distribution.max() / original_distribution.min()
        balanced_ratio = balanced_distribution.max() / balanced_distribution.min()
        
        logger.info(f"Imbalance ratio improvement: {original_ratio:.2f}:1 → {balanced_ratio:.2f}:1")
        
        # Justify the choice
        self._justify_sampling_choice(method, original_ratio, balanced_ratio)
    
    def _justify_sampling_choice(self, method: str, original_ratio: float, balanced_ratio: float) -> None:
        """Justify the choice of sampling method."""
        logger.info(f"Justification for using {method.upper()}:")
        
        if method.lower() == 'smote':
            logger.info("  - SMOTE creates synthetic samples for minority class")
            logger.info("  - Preserves original data distribution")
            logger.info("  - Effective for moderate class imbalance")
            logger.info("  - Maintains feature relationships")
        elif method.lower() == 'adasyn':
            logger.info("  - ADASYN focuses on harder-to-learn samples")
            logger.info("  - Adaptive synthetic sampling")
            logger.info("  - Better for complex decision boundaries")
        elif method.lower() == 'random_under':
            logger.info("  - Random undersampling reduces majority class")
            logger.info("  - Simple and computationally efficient")
            logger.info("  - May lose important information")
        elif method.lower() in ['smoteenn', 'smotetomek']:
            logger.info("  - Hybrid approach combining SMOTE with cleaning")
            logger.info("  - Removes noisy samples")
            logger.info("  - Better quality synthetic samples")
    
    def create_preprocessing_pipeline(self) -> None:
        """Create preprocessing pipeline for numerical and categorical features."""
        logger.info("Creating preprocessing pipeline...")
        
        feature_types = self.identify_feature_types()
        
        # Numerical features preprocessing
        numerical_features = feature_types['numerical']
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Categorical features preprocessing (low cardinality)
        categorical_features = feature_types['categorical']
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # High cardinality features preprocessing (use custom label encoding)
        high_cardinality_features = feature_types['high_cardinality']
        if high_cardinality_features:
            high_cardinality_transformer = HighCardinalityEncoder()
        else:
            high_cardinality_transformer = None
        
        # Combine transformers
        transformers = [
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
        
        # Add high cardinality transformer if there are such features
        if high_cardinality_features and high_cardinality_transformer is not None:
            transformers.append(('high_card', high_cardinality_transformer, high_cardinality_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        logger.info(f"Preprocessing pipeline created with:")
        logger.info(f"  Numerical features: {len(numerical_features)}")
        logger.info(f"  Categorical features: {len(categorical_features)}")
        logger.info(f"  High cardinality features: {len(high_cardinality_features)}")
    
    def fit_transform_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        logger.info("Fitting and transforming data...")
        
        # Fit preprocessor on training data
        X_train_transformed = self.preprocessor.fit_transform(self.X_train_balanced)
        X_test_transformed = self.preprocessor.transform(self.X_test)
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        logger.info(f"Data transformation completed:")
        logger.info(f"  Training shape: {X_train_transformed.shape}")
        logger.info(f"  Testing shape: {X_test_transformed.shape}")
        logger.info(f"  Total features: {X_train_transformed.shape[1]}")
        
        return X_train_transformed, X_test_transformed, self.y_train_balanced, self.y_test
    
    def _get_feature_names(self) -> List[str]:
       
        feature_names = []
        
        # Numerical features
        if 'num' in self.preprocessor.named_transformers_:
            numerical_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(numerical_features)
        
        # Categorical features
        if 'cat' in self.preprocessor.named_transformers_:
            categorical_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(categorical_features)
        
        # High cardinality features (label encoded - keep original names)
        if 'high_card' in self.preprocessor.named_transformers_:
            high_card_features = self.preprocessor.named_transformers_['high_card'].get_feature_names_out()
            feature_names.extend(high_card_features)
        
        return feature_names.tolist()
    
    def save_transformed_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> None:
        """Save transformed data to files."""
        logger.info("Saving transformed data...")
        
        # Save as numpy arrays
        np.save(os.path.join(self.output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'y_test.npy'), y_test)
        
        # Save preprocessor
        with open(os.path.join(self.output_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save feature names
        feature_names = self._get_feature_names()
        with open(os.path.join(self.output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Save metadata
        metadata = {
            'original_shape': self.data.shape,
            'training_shape': X_train.shape,
            'testing_shape': X_test.shape,
            'feature_names': feature_names,
            'class_distribution_original': self.data['class'].value_counts().to_dict(),
            'class_distribution_training': pd.Series(y_train).value_counts().to_dict(),
            'class_distribution_testing': pd.Series(y_test).value_counts().to_dict()
        }
        
        with open(os.path.join(self.output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Transformed data saved to {self.output_dir}")
    
    def generate_transformation_report(self) -> Dict:
        """Generate a comprehensive transformation report."""
        logger.info("Generating transformation report...")
        
        report = {
            'data_info': {
                'original_samples': len(self.data),
                'training_samples': len(self.y_train_balanced),
                'testing_samples': len(self.y_test),
                'total_features': self.X_train_balanced.shape[1]
            },
            'class_distribution': {
                'original': self.data['class'].value_counts().to_dict(),
                'training_balanced': pd.Series(self.y_train_balanced).value_counts().to_dict(),
                'testing': self.y_test.value_counts().to_dict()
            },
            'feature_types': self.identify_feature_types(),
            'preprocessing': {
                'scaler_type': 'StandardScaler',
                'categorical_encoding': 'OneHotEncoder',
                'sampling_method': 'SMOTE',
                'pipeline_order': 'Encode -> SMOTE -> Scale'
            }
        }
        
        # Save report
        import json
        with open(os.path.join(self.output_dir, 'transformation_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Transformation report saved to {self.output_dir}/transformation_report.json")
        return report
    
    def run_full_transformation(self, test_size: float = 0.2, 
                              sampling_method: str = 'smote',
                              random_state: int = 42,
                              memory_efficient: bool = True) -> Dict:
        logger.info("Starting full data transformation pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze class distribution
        class_analysis = self.analyze_class_distribution()
        self.plot_class_distribution()
        
        # Step 3: Split data
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Step 4: Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Step 5: Encode categorical features before SMOTE
        logger.info("Encoding categorical features before SMOTE...")
        
        if memory_efficient:
            # Memory-efficient encoding
            try:
                X_train_encoded = self._encode_memory_efficient(self.X_train, fit=True)
                X_test_encoded = self._encode_memory_efficient(self.X_test, fit=False)
                logger.info("Memory-efficient encoding completed successfully")
            except Exception as e:
                logger.warning(f"Memory-efficient encoding failed: {str(e)}")
                logger.info("Falling back to standard encoding...")
                X_train_encoded = self.preprocessor.fit_transform(self.X_train)
                X_test_encoded = self.preprocessor.transform(self.X_test)
        else:
            # Standard encoding
            X_train_encoded = self.preprocessor.fit_transform(self.X_train)
            X_test_encoded = self.preprocessor.transform(self.X_test)
        
        # Step 6: Handle class imbalance on encoded data
        self.handle_class_imbalance_encoded(X_train_encoded, self.y_train, method=sampling_method, random_state=random_state)
        
        # Step 7: Save transformed data
        self.save_transformed_data(self.X_train_balanced, X_test_encoded, 
                                 self.y_train_balanced, self.y_test)
        
        # Step 8: Generate report
        report = self.generate_transformation_report()
        
        logger.info("Data transformation pipeline completed successfully!")
        
        return {
            'X_train': self.X_train_balanced,
            'X_test': X_test_encoded,
            'y_train': self.y_train_balanced,
            'y_test': self.y_test,
            'preprocessor': self.preprocessor,
            'report': report
        }
    
    def _encode_memory_efficient(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
       
        logger.info("Using memory-efficient encoding...")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Fit mode: {fit}")
        
        feature_types = self.identify_feature_types()
        logger.info(f"Feature types: {feature_types}")
        
        X_encoded = X.copy()
        
        # Handle numerical features (scale them)
        if feature_types['numerical']:
            if fit:
                self.scaler = StandardScaler()
                X_encoded[feature_types['numerical']] = self.scaler.fit_transform(X[feature_types['numerical']])
            else:
                X_encoded[feature_types['numerical']] = self.scaler.transform(X[feature_types['numerical']])
        
        # Handle categorical features (one-hot encode low cardinality)
        for col in feature_types['categorical']:
            if col in X.columns:  # Check if column exists
                if fit:
                    # Create one-hot encoding
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    if not dummies.empty:
                        X_encoded = pd.concat([X_encoded, dummies], axis=1)
                        # Store categorical column information for test data
                        if not hasattr(self, 'categorical_columns'):
                            self.categorical_columns = {}
                        self.categorical_columns[col] = dummies.columns.tolist()
                    X_encoded.drop(columns=[col], inplace=True)
                else:
                    # For test data, handle unseen categories
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    if not dummies.empty:
                        # Ensure all expected columns exist
                        if hasattr(self, 'categorical_columns'):
                            for expected_col in self.categorical_columns.get(col, []):
                                if expected_col not in dummies.columns:
                                    dummies[expected_col] = 0
                        X_encoded = pd.concat([X_encoded, dummies], axis=1)
                    X_encoded.drop(columns=[col], inplace=True)
        
        # Handle high cardinality features (label encode)
        for col in feature_types['high_cardinality']:
            if col in X.columns:  # Check if column exists
                if fit:
                    le = LabelEncoder()
                    # Handle missing values
                    X_encoded[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
                    self.label_encoders = getattr(self, 'label_encoders', {})
                    self.label_encoders[col] = le
                else:
                    # Handle unseen categories in test data
                    le = self.label_encoders[col]
                    # Map unseen categories to -1
                    X_encoded[col] = X[col].astype(str).fillna('missing').map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Ensure we have a valid numpy array
        if X_encoded.empty:
            logger.warning("Encoded dataframe is empty, creating zero array")
            return np.zeros((X.shape[0], 1))
        
        return X_encoded.values


