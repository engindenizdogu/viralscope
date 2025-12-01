import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training and evaluation for YouTube trend prediction.
    
    Trains multiple classifiers and evaluates their performance
    on the stratified sample with engineered features.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize ModelTrainer.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_train_test_split(self, X, y):
        """
        Split data into training and testing sets with stratification.
        """
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {len(X_train):,}")
        print(f"Testing set size: {len(X_test):,}")
        print(f"Training class distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"Testing class distribution:\n{pd.Series(y_test).value_counts()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier.
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*70)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        print(f"Random Forest training completed in {duration:.2f} seconds")
        self.models['random_forest'] = rf_model
        
        return rf_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """
        Train Gradient Boosting classifier.
        """
        print("\n" + "="*70)
        print("TRAINING GRADIENT BOOSTING CLASSIFIER")
        print("="*70)
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state,
            verbose=1
        )
        
        start_time = time.time()
        gb_model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        print(f"Gradient Boosting training completed in {duration:.2f} seconds")
        self.models['gradient_boosting'] = gb_model
        
        return gb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance and print metrics.
        """
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def plot_feature_importance(self, model, feature_names, model_name, output_path, top_n=20):
        """
        Plot and save feature importance.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                  fontsize=14, fontweight='bold')
        plt.bar(range(len(indices)), importances[indices], color='steelblue')
        plt.xticks(range(len(indices)), 
                   [feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved: {output_path}")
    
    def plot_confusion_matrix(self, cm, model_name, output_path):
        """
        Plot and save confusion matrix heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Not Successful', 'Successful'],
                    yticklabels=['Not Successful', 'Successful'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved: {output_path}")
    
    def save_models(self, output_dir='models'):
        """
        Save trained models and scaler to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {scaler_path}")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {model_path}")
    
    def run_training_pipeline(self, X, y, feature_names, output_dir='models'):
        """
        Execute complete model training and evaluation pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature column names
            output_dir: Directory to save models and plots
            
        Returns:
            Dictionary with trained models and evaluation results
        """
        print("="*70)
        print("MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(X, y)
        
        # Train models
        rf_model = self.train_random_forest(X_train, y_train)
        gb_model = self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        gb_results = self.evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        
        # Create output directories
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot feature importance
        self.plot_feature_importance(
            rf_model, feature_names, "Random Forest",
            output_path=os.path.join(plots_dir, 'rf_feature_importance.png')
        )
        self.plot_feature_importance(
            gb_model, feature_names, "Gradient Boosting",
            output_path=os.path.join(plots_dir, 'gb_feature_importance.png')
        )
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            rf_results['confusion_matrix'], "Random Forest",
            output_path=os.path.join(plots_dir, 'rf_confusion_matrix.png')
        )
        self.plot_confusion_matrix(
            gb_results['confusion_matrix'], "Gradient Boosting",
            output_path=os.path.join(plots_dir, 'gb_confusion_matrix.png')
        )
        
        # Save models
        self.save_models(output_dir)
        
        print("\n" + "="*70)
        print("MODEL TRAINING PIPELINE COMPLETED")
        print("="*70)
        
        return {
            'models': self.models,
            'rf_results': rf_results,
            'gb_results': gb_results,
            'X_test': X_test,
            'y_test': y_test
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    
    print("="*70)
    print("STANDALONE MODEL TRAINING")
    print("="*70)
    
    # Load engineered features
    input_path = 'SampleData/data.csv.gz'
    output_dir = 'models'
    
    print(f"\nLoading engineered features from: {input_path}")
    df = pd.read_csv(input_path, compression='gzip', low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Prepare features and target
    print("\nPreparing features and target variable...")
    y = df['is_successful']
    X = df.drop(columns=['is_successful'])
    feature_names = list(X.columns)
    
    # Handle missing and infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")
    
    # Initialize and run model trainer
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    
    results = trainer.run_training_pipeline(
        X=X,
        y=y,
        feature_names=feature_names,
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"Models saved to: {output_dir}/")
    print(f"Plots saved to: {output_dir}/plots/")
