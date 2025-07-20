
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EDAAnalyzer:
    
    def __init__(self):
        self.fraud_data: Optional[pd.DataFrame] = None
        self.credit_card_data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict = {}
        
    def load_data(self, fraud_data: pd.DataFrame, credit_card_data: Optional[pd.DataFrame] = None):
    
        self.fraud_data = fraud_data
        self.credit_card_data = credit_card_data
        logger.info("Data loaded for EDA analysis")
    
    def _detect_target_column(self, df: pd.DataFrame) -> str:
     
        if 'class' in df.columns:
            return 'class'
        elif 'Class' in df.columns:
            return 'Class'
        else:
            raise ValueError("No target column found. Expected 'class' or 'Class'")
    
    def _detect_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the amount column name"""
        amount_columns = ['amount', 'Amount', 'purchase_value', 'purchase_value']
        for col in amount_columns:
            if col in df.columns:
                return col
        return None
    
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the time column name"""
        time_columns = ['Time', 'purchase_time', 'signup_time']
        for col in time_columns:
            if col in df.columns:
                return col
        return None
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp columns"""
        df_copy = df.copy()
        
        # Detect time column
        time_col = self._detect_time_column(df_copy)
        if time_col is None:
            logger.warning("No time column found. Time-based analysis will be skipped.")
            return df_copy
        
        try:
            # Convert to datetime if not already
            if df_copy[time_col].dtype == 'object':
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            
            # Create time features
            df_copy['hour_of_day'] = df_copy[time_col].dt.hour
            df_copy['day_of_week'] = df_copy[time_col].dt.day_name()
            df_copy['day_of_month'] = df_copy[time_col].dt.day
            df_copy['month'] = df_copy[time_col].dt.month
            df_copy['year'] = df_copy[time_col].dt.year
            
            logger.info(f"Created time features from column: {time_col}")
            
        except Exception as e:
            logger.warning(f"Could not create time features: {e}")
        
        return df_copy
        
    def analyze_class_distribution(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
     
        logger.info("Analyzing class distribution...")
        
        # Detect target column if not provided
        if target_col is None:
            target_col = self._detect_target_column(df)
        
        class_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        # Handle potential None values safely
        fraud_count = class_counts.get(1, 0) or 0
        legitimate_count = class_counts.get(0, 0) or 0
        
        analysis = {
            'class_counts': class_counts.to_dict(),
            'total_samples': total_samples,
            'fraud_percentage': (fraud_count / total_samples) * 100,
            'legitimate_percentage': (legitimate_count / total_samples) * 100,
            'imbalance_ratio': class_counts.min() / class_counts.max(),
            'fraud_count': fraud_count,
            'legitimate_count': legitimate_count
        }
        
        logger.info(f"Class distribution analysis completed")
        logger.info(f"Fraud percentage: {analysis['fraud_percentage']:.2f}%")
        logger.info(f"Imbalance ratio: {analysis['imbalance_ratio']:.4f}")
        
        return analysis
    
    def plot_class_distribution(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                              save_path: Optional[str] = None) -> None:
    
        # Detect target column if not provided
        if target_col is None:
            target_col = self._detect_target_column(df)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=df, x=target_col, ax=ax1, palette=['lightblue', 'red'])
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        
        # Add count labels on bars
        for i, v in enumerate(df[target_col].value_counts().sort_index()):
            ax1.text(i, v + v*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        class_counts = df[target_col].value_counts()
        colors = ['lightblue', 'red']
        labels = ['Legitimate', 'Fraud']
        
        ax2.pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90)
        ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def analyze_fraud_rate_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze fraud rate by hour of day"""
        logger.info("Analyzing fraud rate by hour...")
        
        # Create time features if not present
        df_with_time = self._create_time_features(df)
        
        if 'hour_of_day' not in df_with_time.columns:
            logger.error("Required column 'hour_of_day' not found and could not be created")
            return pd.DataFrame()
        
        target_col = self._detect_target_column(df_with_time)
        
        # Calculate fraud rate by hour
        fraud_by_hour = df_with_time.groupby('hour_of_day')[target_col].agg(['count', 'sum']).reset_index()
        fraud_by_hour.columns = ['hour_of_day', 'total_transactions', 'fraud_count']
        fraud_by_hour['fraud_rate'] = (fraud_by_hour['fraud_count'] / fraud_by_hour['total_transactions']) * 100
        
        # Identify peak fraud hours
        peak_hours = fraud_by_hour.nlargest(3, 'fraud_rate')
        
        logger.info("Fraud rate by hour analysis completed")
        logger.info(f"Peak fraud hours: {peak_hours[['hour_of_day', 'fraud_rate']].to_dict('records')}")
        
        return fraud_by_hour
    
    def plot_fraud_rate_by_hour(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot fraud rate by hour"""
        fraud_by_hour = self.analyze_fraud_rate_by_hour(df)
        
        if fraud_by_hour.empty:
            logger.warning("No data available for fraud rate by hour analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Fraud rate line plot
        sns.lineplot(data=fraud_by_hour, x='hour_of_day', y='fraud_rate', 
                    marker='o', linewidth=2, markersize=8, ax=ax1)
        ax1.set_title("Fraud Rate by Hour", fontsize=16, fontweight='bold')
        ax1.set_xlabel("Hour of Day", fontsize=12)
        ax1.set_ylabel("Fraud Rate (%)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24))
        
        # Transaction volume by hour
        sns.lineplot(data=fraud_by_hour, x='hour_of_day', y='total_transactions', 
                    marker='s', linewidth=2, markersize=8, ax=ax2, color='green')
        ax2.set_title("Transaction Volume by Hour", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Hour of Day", fontsize=12)
        ax2.set_ylabel("Total Transactions", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fraud rate by hour plot saved to {save_path}")
        
        plt.show()
    
    def analyze_fraud_rate_by_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze fraud rate by day of week"""
        logger.info("Analyzing fraud rate by day of week...")
        
        # Create time features if not present
        df_with_time = self._create_time_features(df)
        
        if 'day_of_week' not in df_with_time.columns:
            logger.error("Required column 'day_of_week' not found and could not be created")
            return pd.DataFrame()
        
        target_col = self._detect_target_column(df_with_time)
        
        # Calculate fraud rate by day
        fraud_by_day = df_with_time.groupby('day_of_week')[target_col].agg(['count', 'sum']).reset_index()
        fraud_by_day.columns = ['day_of_week', 'total_transactions', 'fraud_count']
        fraud_by_day['fraud_rate'] = (fraud_by_day['fraud_count'] / fraud_by_day['total_transactions']) * 100
        
        # Sort by day of week (Monday=0, Sunday=6)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fraud_by_day['day_order'] = fraud_by_day['day_of_week'].map({day: i for i, day in enumerate(day_order)})
        fraud_by_day = fraud_by_day.sort_values('day_order')
        
        logger.info("Fraud rate by day analysis completed")
        
        return fraud_by_day
    
    def plot_fraud_rate_by_day(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot fraud rate by day of week"""
        fraud_by_day = self.analyze_fraud_rate_by_day(df)
        
        if fraud_by_day.empty:
            logger.warning("No data available for fraud rate by day analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Fraud rate bar plot
        sns.barplot(data=fraud_by_day, x='day_of_week', y='fraud_rate', ax=ax1, palette='viridis')
        ax1.set_title("Fraud Rate by Day of Week", fontsize=16, fontweight='bold')
        ax1.set_xlabel("Day of Week", fontsize=12)
        ax1.set_ylabel("Fraud Rate (%)", fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Transaction volume by day
        sns.barplot(data=fraud_by_day, x='day_of_week', y='total_transactions', ax=ax2, palette='plasma')
        ax2.set_title("Transaction Volume by Day of Week", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Day of Week", fontsize=12)
        ax2.set_ylabel("Total Transactions", fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fraud rate by day plot saved to {save_path}")
        
        plt.show()
    
    def analyze_amount_distribution(self, df: pd.DataFrame, amount_col: Optional[str] = None) -> Dict:
        """Analyze amount distribution"""
        logger.info("Analyzing amount distribution...")
        
        # Detect amount column if not provided
        if amount_col is None:
            amount_col = self._detect_amount_column(df)
        
        if amount_col is None:
            logger.error("No amount column found")
            return {}
        
        target_col = self._detect_target_column(df)
        
        # Separate fraud and legitimate transactions
        fraud_amounts = df[df[target_col] == 1][amount_col]
        legitimate_amounts = df[df[target_col] == 0][amount_col]
        
        analysis = {
            'fraud_amount_stats': {
                'mean': fraud_amounts.mean(),
                'median': fraud_amounts.median(),
                'std': fraud_amounts.std(),
                'min': fraud_amounts.min(),
                'max': fraud_amounts.max(),
                'q25': fraud_amounts.quantile(0.25),
                'q75': fraud_amounts.quantile(0.75)
            },
            'legitimate_amount_stats': {
                'mean': legitimate_amounts.mean(),
                'median': legitimate_amounts.median(),
                'std': legitimate_amounts.std(),
                'min': legitimate_amounts.min(),
                'max': legitimate_amounts.max(),
                'q25': legitimate_amounts.quantile(0.25),
                'q75': legitimate_amounts.quantile(0.75)
            },
            'amount_difference': {
                'mean_diff': fraud_amounts.mean() - legitimate_amounts.mean(),
                'median_diff': fraud_amounts.median() - legitimate_amounts.median()
            }
        }
        
        logger.info("Amount distribution analysis completed")
        logger.info(f"Fraud mean amount: ${analysis['fraud_amount_stats']['mean']:.2f}")
        logger.info(f"Legitimate mean amount: ${analysis['legitimate_amount_stats']['mean']:.2f}")
        
        return analysis
    
    def plot_amount_distribution(self, df: pd.DataFrame, amount_col: Optional[str] = None, 
                               save_path: Optional[str] = None) -> None:
        """Plot amount distribution"""
        # Detect amount column if not provided
        if amount_col is None:
            amount_col = self._detect_amount_column(df)
        
        if amount_col is None:
            logger.error("No amount column found")
            return
        
        target_col = self._detect_target_column(df)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Separate data
        fraud_data = df[df[target_col] == 1]
        legitimate_data = df[df[target_col] == 0]
        
        # Histogram comparison
        ax1.hist(legitimate_data[amount_col], bins=50, alpha=0.7, label='Legitimate', color='lightblue')
        ax1.hist(fraud_data[amount_col], bins=50, alpha=0.7, label='Fraud', color='red')
        ax1.set_title('Amount Distribution Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Amount ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Box plot
        df.boxplot(column=amount_col, by=target_col, ax=ax2)
        ax2.set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
        ax2.set_ylabel('Amount ($)', fontsize=12)
        
        # Violin plot
        sns.violinplot(data=df, x=target_col, y=amount_col, ax=ax3)
        ax3.set_title('Amount Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
        ax3.set_ylabel('Amount ($)', fontsize=12)
        
        # Cumulative distribution
        fraud_sorted = np.sort(fraud_data[amount_col])
        legitimate_sorted = np.sort(legitimate_data[amount_col])
        
        ax4.plot(fraud_sorted, np.arange(1, len(fraud_sorted) + 1) / len(fraud_sorted), 
                label='Fraud', color='red')
        ax4.plot(legitimate_sorted, np.arange(1, len(legitimate_sorted) + 1) / len(legitimate_sorted), 
                label='Legitimate', color='lightblue')
        ax4.set_title('Cumulative Amount Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Amount ($)', fontsize=12)
        ax4.set_ylabel('Cumulative Probability', fontsize=12)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Amount distribution plot saved to {save_path}")
        
        plt.show()
    
    def analyze_geographic_patterns(self, df: pd.DataFrame, country_col: str = 'country') -> Dict:
        """Analyze geographic fraud patterns"""
        logger.info("Analyzing geographic fraud patterns...")
        
        target_col = self._detect_target_column(df)
        
        if country_col not in df.columns:
            logger.error(f"Required column '{country_col}' not found")
            return {}
        
        # Calculate fraud rate by country
        geo_analysis = df.groupby(country_col)[target_col].agg(['count', 'sum']).reset_index()
        geo_analysis.columns = [country_col, 'total_transactions', 'fraud_count']
        geo_analysis['fraud_rate'] = (geo_analysis['fraud_count'] / geo_analysis['total_transactions']) * 100
        
        # Sort by fraud rate (descending)
        geo_analysis = geo_analysis.sort_values('fraud_rate', ascending=False)
        
        # Get top fraud countries
        top_fraud_countries = geo_analysis.head(10)
        
        analysis = {
            'geo_analysis': geo_analysis,
            'top_fraud_countries': top_fraud_countries,
            'total_countries': len(geo_analysis),
            'high_risk_countries': len(geo_analysis[geo_analysis['fraud_rate'] > geo_analysis['fraud_rate'].mean()])
        }
        
        logger.info("Geographic analysis completed")
        logger.info(f"Top fraud country: {top_fraud_countries.iloc[0][country_col]} "
                   f"({top_fraud_countries.iloc[0]['fraud_rate']:.2f}% fraud rate)")
        
        return analysis
    
    def plot_geographic_patterns(self, df: pd.DataFrame, country_col: str = 'country', 
                               save_path: Optional[str] = None) -> None:
      
        geo_analysis = self.analyze_geographic_patterns(df, country_col)
        
        if not geo_analysis:
            logger.warning("No geographic data available for analysis")
            return
        
        top_countries = geo_analysis['top_fraud_countries']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top fraud countries bar plot
        sns.barplot(data=top_countries, x=country_col, y='fraud_rate', ax=ax1, palette='Reds')
        ax1.set_title('Top 10 Countries by Fraud Rate', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Country', fontsize=12)
        ax1.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Transaction volume by country
        sns.barplot(data=top_countries, x=country_col, y='total_transactions', ax=ax2, palette='Blues')
        ax2.set_title('Transaction Volume by Country (Top 10 Fraud)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Country', fontsize=12)
        ax2.set_ylabel('Total Transactions', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Geographic patterns plot saved to {save_path}")
        
        plt.show()
    
    def analyze_feature_correlations(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
     
        logger.info("Analyzing feature correlations...")
        
        # Detect target column if not provided
        if target_col is None:
            target_col = self._detect_target_column(df)
        
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in numerical_cols:
            logger.error(f"Target column '{target_col}' not found in numerical columns")
            return pd.Series(dtype=float)
        
        # Calculate correlations with target using only numerical columns
        correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
        
        # Remove target column from correlations
        correlations = correlations.drop(target_col)
        
        # Get top positive and negative correlations
        top_positive = correlations.head(10)
        top_negative = correlations.tail(10)
        
        logger.info("Feature correlation analysis completed")
        if len(correlations) > 0:
            logger.info(f"Top positive correlation: {top_positive.index[0]} ({top_positive.iloc[0]:.4f})")
            logger.info(f"Top negative correlation: {top_negative.index[-1]} ({top_negative.iloc[-1]:.4f})")
        else:
            logger.warning("No numerical features found for correlation analysis")
        
        return correlations
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                               save_path: Optional[str] = None) -> None:
  
        # Detect target column if not provided
        if target_col is None:
            target_col = self._detect_target_column(df)
        
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in numerical_cols:
            logger.error(f"Target column '{target_col}' not found in numerical columns")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def comprehensive_analysis(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Dict:
    
        logger.info("Starting comprehensive EDA analysis...")
        
        results = {}
        target_col = self._detect_target_column(df)
        
        # 1. Class distribution analysis
        results['class_distribution'] = self.analyze_class_distribution(df, target_col)
        if save_dir:
            self.plot_class_distribution(df, target_col, save_path=f"{save_dir}/class_distribution.png")
        else:
            self.plot_class_distribution(df, target_col)
        
        # 2. Time-based analysis
        df_with_time = self._create_time_features(df)
        if 'hour_of_day' in df_with_time.columns:
            results['fraud_by_hour'] = self.analyze_fraud_rate_by_hour(df)
            if save_dir:
                self.plot_fraud_rate_by_hour(df, save_path=f"{save_dir}/fraud_by_hour.png")
            else:
                self.plot_fraud_rate_by_hour(df)
        
        if 'day_of_week' in df_with_time.columns:
            results['fraud_by_day'] = self.analyze_fraud_rate_by_day(df)
            if save_dir:
                self.plot_fraud_rate_by_day(df, save_path=f"{save_dir}/fraud_by_day.png")
            else:
                self.plot_fraud_rate_by_day(df)
        
        # 3. Amount analysis
        amount_col = self._detect_amount_column(df)
        if amount_col:
            results['amount_analysis'] = self.analyze_amount_distribution(df, amount_col)
            if save_dir:
                self.plot_amount_distribution(df, amount_col, save_path=f"{save_dir}/amount_distribution.png")
            else:
                self.plot_amount_distribution(df, amount_col)
        
        # 4. Geographic analysis
        country_cols = [col for col in df.columns if 'country' in col.lower()]
        if country_cols:
            country_col = country_cols[0]
            results['geographic_analysis'] = self.analyze_geographic_patterns(df, country_col)
            if save_dir:
                self.plot_geographic_patterns(df, country_col, save_path=f"{save_dir}/geographic_patterns.png")
            else:
                self.plot_geographic_patterns(df, country_col)
        
        # 5. Feature correlations
        results['correlations'] = self.analyze_feature_correlations(df, target_col)
        if save_dir:
            self.plot_correlation_heatmap(df, target_col, save_path=f"{save_dir}/correlation_heatmap.png")
        else:
            self.plot_correlation_heatmap(df, target_col)
        
        # 6. Summary statistics
        results['summary_stats'] = {
            'total_transactions': len(df),
            'fraud_transactions': len(df[df[target_col] == 1]),
            'legitimate_transactions': len(df[df[target_col] == 0]),
            'fraud_percentage': (len(df[df[target_col] == 1]) / len(df)) * 100,
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'has_time_features': 'hour_of_day' in df_with_time.columns,
            'has_amount_features': amount_col is not None,
            'has_geographic_features': len(country_cols) > 0
        }
        
        logger.info("Comprehensive EDA analysis completed")
        logger.info(f"Analysis summary: {results['summary_stats']}")
        
        return results
    
    def generate_insights_report(self, analysis_results: Dict) -> str:
     
        logger.info("Generating insights report...")
        
        report = []
        report.append("=" * 80)
        report.append("FRAUD DETECTION - EXPLORATORY DATA ANALYSIS INSIGHTS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary Statistics
        if 'summary_stats' in analysis_results:
            stats = analysis_results['summary_stats']
            report.append("📊 SUMMARY STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Transactions: {stats['total_transactions']:,}")
            report.append(f"Fraud Transactions: {stats['fraud_transactions']:,}")
            report.append(f"Legitimate Transactions: {stats['legitimate_transactions']:,}")
            report.append(f"Fraud Percentage: {stats['fraud_percentage']:.2f}%")
            report.append(f"Numerical Features: {stats['numerical_features']}")
            report.append(f"Categorical Features: {stats['categorical_features']}")
            report.append(f"Has Time Features: {stats['has_time_features']}")
            report.append(f"Has Amount Features: {stats['has_amount_features']}")
            report.append(f"Has Geographic Features: {stats['has_geographic_features']}")
            report.append("")
        
        # Class Distribution Insights
        if 'class_distribution' in analysis_results:
            dist = analysis_results['class_distribution']
            report.append("🎯 CLASS DISTRIBUTION INSIGHTS")
            report.append("-" * 40)
            report.append(f"• Dataset is highly imbalanced with {dist['fraud_percentage']:.2f}% fraud cases")
            report.append(f"• Imbalance ratio: {dist['imbalance_ratio']:.4f}")
            if dist['imbalance_ratio'] < 0.1:
                report.append("• ⚠️  Severe class imbalance detected - consider resampling techniques")
            elif dist['imbalance_ratio'] < 0.3:
                report.append("• ⚠️  Moderate class imbalance detected - consider balanced metrics")
            else:
                report.append("• ✅ Relatively balanced dataset")
            report.append("")
        
        # Time-based Insights
        if 'fraud_by_hour' in analysis_results:
            hour_data = analysis_results['fraud_by_hour']
            peak_hours = hour_data.nlargest(3, 'fraud_rate')
            report.append("⏰ TEMPORAL PATTERNS - HOURLY")
            report.append("-" * 40)
            report.append("Peak fraud hours:")
            for _, row in peak_hours.iterrows():
                report.append(f"  • Hour {int(row['hour_of_day']):02d}:00 - {row['fraud_rate']:.2f}% fraud rate")
            report.append("")
        
        if 'fraud_by_day' in analysis_results:
            day_data = analysis_results['fraud_by_day']
            peak_days = day_data.nlargest(3, 'fraud_rate')
            report.append("📅 TEMPORAL PATTERNS - DAILY")
            report.append("-" * 40)
            report.append("Peak fraud days:")
            for _, row in peak_days.iterrows():
                report.append(f"  • {row['day_of_week']} - {row['fraud_rate']:.2f}% fraud rate")
            report.append("")
        
        # Amount Analysis Insights
        if 'amount_analysis' in analysis_results:
            amount_data = analysis_results['amount_analysis']
            report.append("💰 AMOUNT ANALYSIS INSIGHTS")
            report.append("-" * 40)
            
            fraud_mean = amount_data['fraud_amount_stats']['mean']
            legit_mean = amount_data['legitimate_amount_stats']['mean']
            mean_diff = amount_data['amount_difference']['mean_diff']
            
            report.append(f"• Average fraud amount: ${fraud_mean:.2f}")
            report.append(f"• Average legitimate amount: ${legit_mean:.2f}")
            report.append(f"• Difference: ${mean_diff:.2f}")
            
            if mean_diff > 0:
                report.append("• 💡 Fraud transactions tend to be higher value")
            elif mean_diff < 0:
                report.append("• 💡 Fraud transactions tend to be lower value")
            else:
                report.append("• 💡 No significant difference in transaction amounts")
            report.append("")
        
        # Geographic Insights
        if 'geographic_analysis' in analysis_results:
            geo_data = analysis_results['geographic_analysis']
            top_countries = geo_data['top_fraud_countries']
            report.append("🌍 GEOGRAPHIC INSIGHTS")
            report.append("-" * 40)
            report.append("Top 5 high-risk countries:")
            for i, (_, row) in enumerate(top_countries.head().iterrows(), 1):
                report.append(f"  {i}. {row['country']} - {row['fraud_rate']:.2f}% fraud rate")
            report.append(f"• Total countries analyzed: {geo_data['total_countries']}")
            report.append(f"• High-risk countries: {geo_data['high_risk_countries']}")
            report.append("")
        
        # Feature Correlation Insights
        if 'correlations' in analysis_results:
            corr_data = analysis_results['correlations']
            report.append("🔗 FEATURE CORRELATION INSIGHTS")
            report.append("-" * 40)
            
            # Top positive correlations
            top_positive = corr_data.head(5)
            report.append("Top 5 positive correlations with fraud:")
            for feature, corr in top_positive.items():
                report.append(f"  • {feature}: {corr:.4f}")
            
            # Top negative correlations
            top_negative = corr_data.tail(5)
            report.append("Top 5 negative correlations with fraud:")
            for feature, corr in top_negative.items():
                report.append(f"  • {feature}: {corr:.4f}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Class imbalance recommendations
        if 'class_distribution' in analysis_results:
            dist = analysis_results['class_distribution']
            if dist['imbalance_ratio'] < 0.1:
                report.append("• Use SMOTE, ADASYN, or other resampling techniques")
                report.append("• Consider ensemble methods (Random Forest, XGBoost)")
                report.append("• Use balanced accuracy, F1-score, or AUC-ROC for evaluation")
            elif dist['imbalance_ratio'] < 0.3:
                report.append("• Consider class weights in model training")
                report.append("• Use balanced evaluation metrics")
        
        # Feature engineering recommendations
        if 'correlations' in analysis_results:
            corr_data = analysis_results['correlations']
            strong_corr = corr_data[abs(corr_data) > 0.3]
            if len(strong_corr) > 0:
                report.append("• Focus on features with strong correlations for feature engineering")
                report.append("• Consider interaction terms between highly correlated features")
        
        # Temporal recommendations
        if 'fraud_by_hour' in analysis_results or 'fraud_by_day' in analysis_results:
            report.append("• Include temporal features in model training")
            report.append("• Consider time-based cross-validation")
        
        # General recommendations
        report.append("• Use cross-validation with stratified sampling")
        report.append("• Consider anomaly detection techniques")
        report.append("• Implement real-time monitoring for high-risk patterns")
        report.append("")
        
        # Model suggestions
        report.append("🤖 SUGGESTED MODELS")
        report.append("-" * 40)
        report.append("• Random Forest (handles imbalanced data well)")
        report.append("• XGBoost/LightGBM (good performance on tabular data)")
        report.append("• Isolation Forest (for anomaly detection)")
        report.append("• Autoencoder (for unsupervised fraud detection)")
        report.append("• Ensemble methods (combine multiple models)")
        report.append("")
        
        report.append("=" * 80)
        report.append("Report generated successfully!")
        report.append("=" * 80)
        
        final_report = "\n".join(report)
        logger.info("Insights report generated successfully")
        
        return final_report