import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessedDataMerger:
    """
    Data merger for processed fraud and credit card datasets.
    Combines geolocation-enriched fraud data with credit card transaction data.
    """

    def __init__(self, fraud_processed_path: str, credit_processed_path: str):
        self.fraud_processed_path = fraud_processed_path
        self.credit_processed_path = credit_processed_path
        self.fraud_data = None
        self.credit_data = None
        self.merged_data = None
        
    def load_processed_data(self) -> None:
        """Load the processed fraud and credit card datasets."""
        try:
            logger.info("Loading processed fraud data...")
            self.fraud_data = pd.read_csv(self.fraud_processed_path)
            logger.info(f"Fraud processed data loaded: {self.fraud_data.shape}")
            
            logger.info("Loading processed credit card data...")
            self.credit_data = pd.read_csv(self.credit_processed_path)
            logger.info(f"Credit processed data loaded: {self.credit_data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def analyze_fraud_data(self) -> Dict:
        """Analyze the fraud dataset with geolocation information."""
        if self.fraud_data is None:
            raise ValueError("Fraud data not loaded. Call load_processed_data() first.")
        
        logger.info("Analyzing fraud data with geolocation...")
        
        analysis = {}
        
        # Basic statistics
        analysis['total_records'] = len(self.fraud_data)
        analysis['fraud_records'] = len(self.fraud_data[self.fraud_data['class'] == 1])
        analysis['non_fraud_records'] = len(self.fraud_data[self.fraud_data['class'] == 0])
        analysis['fraud_rate'] = analysis['fraud_records'] / analysis['total_records']
        
        # Country analysis
        country_analysis = self.fraud_data.groupby('country').agg({
            'class': ['count', 'sum']
        }).round(4)
        country_analysis.columns = ['Total_Transactions', 'Fraud_Count']
        country_analysis['Fraud_Rate'] = (country_analysis['Fraud_Count'] / country_analysis['Total_Transactions']).round(4)
        analysis['country_analysis'] = country_analysis.sort_values('Total_Transactions', ascending=False)
        
        # Top countries by transaction volume
        analysis['top_countries'] = country_analysis.head(10)
        
        # High-risk countries (countries with fraud rate > 10%)
        high_risk_countries = country_analysis[country_analysis['Fraud_Rate'] > 0.10]
        analysis['high_risk_countries'] = high_risk_countries
        
        logger.info(f"Fraud rate: {analysis['fraud_rate']:.4f}")
        logger.info(f"Countries found: {len(country_analysis)}")
        logger.info(f"High-risk countries: {len(high_risk_countries)}")
        
        return analysis
    
    def analyze_credit_data(self) -> Dict:
        """Analyze the credit card dataset."""
        if self.credit_data is None:
            raise ValueError("Credit data not loaded. Call load_processed_data() first.")
        
        logger.info("Analyzing credit card data...")
        
        analysis = {}
        
        # Basic statistics
        analysis['total_records'] = len(self.credit_data)
        analysis['fraud_records'] = len(self.credit_data[self.credit_data['Class'] == 1])
        analysis['non_fraud_records'] = len(self.credit_data[self.credit_data['Class'] == 0])
        analysis['fraud_rate'] = analysis['fraud_records'] / analysis['total_records']
        
        # Amount analysis
        analysis['amount_stats'] = self.credit_data['Amount'].describe()
        
        # Fraud by amount ranges
        amount_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        amount_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
        self.credit_data['amount_category'] = pd.cut(self.credit_data['Amount'], bins=amount_bins, labels=amount_labels)
        
        amount_fraud_analysis = self.credit_data.groupby('amount_category').agg({
            'Class': ['count', 'sum']
        }).round(4)
        amount_fraud_analysis.columns = ['Total_Transactions', 'Fraud_Count']
        amount_fraud_analysis['Fraud_Rate'] = (amount_fraud_analysis['Fraud_Count'] / amount_fraud_analysis['Total_Transactions']).round(4)
        analysis['amount_fraud_analysis'] = amount_fraud_analysis
        
        logger.info(f"Credit card fraud rate: {analysis['fraud_rate']:.6f}")
        logger.info(f"Average transaction amount: ${analysis['amount_stats']['mean']:.2f}")
        
        return analysis
    
    def create_comprehensive_analysis(self) -> Dict:
        """Create a comprehensive analysis combining both datasets."""
        logger.info("Creating comprehensive analysis...")
        
        # Load data if not already loaded
        if self.fraud_data is None or self.credit_data is None:
            self.load_processed_data()
        
        # Analyze both datasets
        fraud_analysis = self.analyze_fraud_data()
        credit_analysis = self.analyze_credit_data()
        
        # Create comprehensive summary
        comprehensive_analysis = {
            'fraud_dataset': fraud_analysis,
            'credit_dataset': credit_analysis,
            'summary': {
                'total_fraud_records': fraud_analysis['fraud_records'] + credit_analysis['fraud_records'],
                'total_transactions': fraud_analysis['total_records'] + credit_analysis['total_records'],
                'overall_fraud_rate': (fraud_analysis['fraud_records'] + credit_analysis['fraud_records']) / 
                                    (fraud_analysis['total_records'] + credit_analysis['total_records']),
                'fraud_dataset_fraud_rate': fraud_analysis['fraud_rate'],
                'credit_dataset_fraud_rate': credit_analysis['fraud_rate']
            }
        }
        
        logger.info("Comprehensive analysis completed!")
        return comprehensive_analysis
    
    def get_geolocation_summary(self) -> pd.DataFrame:
        """Get a detailed geolocation summary for fraud analysis."""
        if self.fraud_data is None:
            raise ValueError("Fraud data not loaded. Call load_processed_data() first.")
        
        logger.info("Creating geolocation summary...")
        
        # Create detailed geolocation analysis
        geolocation_summary = self.fraud_data.groupby('country').agg({
            'class': ['count', 'sum'],
            'purchase_value': ['mean', 'sum'],
            'user_id': 'nunique'
        }).round(4)
        
        geolocation_summary.columns = [
            'Total_Transactions', 'Fraud_Count', 
            'Avg_Purchase_Value', 'Total_Purchase_Value',
            'Unique_Users'
        ]
        
        geolocation_summary['Fraud_Rate'] = (geolocation_summary['Fraud_Count'] / geolocation_summary['Total_Transactions']).round(4)
        geolocation_summary['Avg_Transaction_Value'] = (geolocation_summary['Total_Purchase_Value'] / geolocation_summary['Total_Transactions']).round(2)
        
        # Sort by total transactions
        geolocation_summary = geolocation_summary.sort_values('Total_Transactions', ascending=False)
        
        return geolocation_summary
    
    def save_analysis_results(self, output_path: str, analysis: Dict) -> None:
        """Save analysis results to files."""
        try:
            logger.info(f"Saving analysis results to {output_path}...")
            
            # Create output directory if it doesn't exist
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            
            # Save geolocation summary
            geolocation_summary = self.get_geolocation_summary()
            geolocation_path = output_path.replace('.csv', '_geolocation_summary.csv')
            geolocation_summary.to_csv(geolocation_path)
            logger.info(f"Geolocation summary saved to {geolocation_path}")
            
            # Save country analysis
            country_analysis = analysis['fraud_dataset']['country_analysis']
            country_path = output_path.replace('.csv', '_country_analysis.csv')
            country_analysis.to_csv(country_path)
            logger.info(f"Country analysis saved to {country_path}")
            
            # Save comprehensive summary
            summary_df = pd.DataFrame([analysis['summary']])
            summary_path = output_path.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise
    
    def print_analysis_summary(self, analysis: Dict) -> None:
        """Print a formatted summary of the analysis."""
        print("\n" + "="*60)
    
    def print_key_insights(self, analysis: Dict) -> None:
        """Print key insights from the analysis."""
        print("\nKEY INSIGHTS:")
        print("="*40)
        
        # Fraud rate comparison
        fraud_rate = analysis['fraud_dataset']['fraud_rate']
        credit_rate = analysis['credit_dataset']['fraud_rate']
        print(f"• Fraud dataset fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
        print(f"• Credit card dataset fraud rate: {credit_rate:.6f} ({credit_rate*100:.4f}%)")
        
        # Country insights
        top_countries = analysis['fraud_dataset']['top_countries']
        print(f"\n• Top country by transactions: {top_countries.index[0]} ({top_countries.iloc[0]['Total_Transactions']:,} transactions)")
        
        high_risk_countries = analysis['fraud_dataset']['high_risk_countries']
        if not high_risk_countries.empty:
            worst_country = high_risk_countries.iloc[0]
            print(f"• Highest fraud rate country: {high_risk_countries.index[0]} ({worst_country['Fraud_Rate']:.4f} fraud rate)")
        else:
            print("• No high-risk countries found (fraud rate > 10%)")
        
        # Credit card insights
        credit_stats = analysis['credit_dataset']['amount_stats']
        print(f"\n• Average credit card transaction: ${credit_stats['mean']:.2f}")
        print(f"• Credit card transactions analyzed: {analysis['credit_dataset']['total_records']:,}")
        
        # Additional insights
        print(f"\n• Total combined transactions: {analysis['summary']['total_transactions']:,}")
        print(f"• Overall fraud rate: {analysis['summary']['overall_fraud_rate']:.4f} ({analysis['summary']['overall_fraud_rate']*100:.2f}%)")
        
        print("="*40)
        print("COMPREHENSIVE FRAUD DETECTION ANALYSIS")
        print("="*60)
        
        # Overall summary
        summary = analysis['summary']
        print(f"\nOVERALL SUMMARY:")
        print(f"Total Transactions: {summary['total_transactions']:,}")
        print(f"Total Fraud Records: {summary['total_fraud_records']:,}")
        print(f"Overall Fraud Rate: {summary['overall_fraud_rate']:.4f}")
        
        # Fraud dataset summary
        fraud_summary = analysis['fraud_dataset']
        print(f"\nFRAUD DATASET (with Geolocation):")
        print(f"Records: {fraud_summary['total_records']:,}")
        print(f"Fraud Rate: {fraud_summary['fraud_rate']:.4f}")
        print(f"Countries: {len(fraud_summary['country_analysis'])}")
        
        # Credit dataset summary
        credit_summary = analysis['credit_dataset']
        print(f"\nCREDIT CARD DATASET:")
        print(f"Records: {credit_summary['total_records']:,}")
        print(f"Fraud Rate: {credit_summary['fraud_rate']:.6f}")
        
        # Top countries
        print(f"\nTOP 5 COUNTRIES BY TRANSACTION VOLUME:")
        top_countries = fraud_summary['top_countries'].head(5)
        for country, row in top_countries.iterrows():
            print(f"  {country}: {row['Total_Transactions']:,} transactions, {row['Fraud_Rate']:.4f} fraud rate")
        
        # High-risk countries
        high_risk = fraud_summary['high_risk_countries']
        if not high_risk.empty:
            print(f"\nHIGH-RISK COUNTRIES (Fraud Rate > 10%):")
            for country, row in high_risk.iterrows():
                print(f"  {country}: {row['Fraud_Rate']:.4f} fraud rate ({row['Total_Transactions']:,} transactions)")
        
        print("\n" + "="*60) 