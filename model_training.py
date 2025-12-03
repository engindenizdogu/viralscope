import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    
    def __init__(self, test_size=0.2, random_state=42, n_jobs=-1):
        """
        Initialize ModelTrainer.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.evaluation_results = []
        self.best_params = {}
        
    def prepare_train_test_split(self, X, y):
        """
        Split data into training and testing sets with stratification.
        Note: Features are already scaled in feature_engineering.py
        """
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Training set size: {len(X_train):,}")
        print(f"Testing set size: {len(X_test):,}")
        print(f"Training class distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"Testing class distribution:\n{pd.Series(y_test).value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_model_configs(self):
        """
        Get model configurations with base estimators and hyperparameter grids.
        
        Returns:
            Dictionary mapping model names to tuples of (estimator, param_grid)
        """
        configs = {
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, 
                    random_state=self.random_state, 
                    n_jobs=self.n_jobs
                ),
                {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [10],
                    'min_samples_leaf': [5],
                    'class_weight': ['balanced']
                }
            ),
            "DecisionTree": (
                DecisionTreeClassifier(
                    random_state=self.random_state
                ),
                {
                    'max_depth': [10],
                    'min_samples_split': [10],
                    'min_samples_leaf': [5],
                    'criterion': ['entropy'] #['gini', 'entropy'],
                    #'class_weight': ['balanced', None]
                }
            ),
            "LinearSVC": (
                LinearSVC(
                    max_iter=20000, 
                    random_state=self.random_state
                ),
                {
                    'C': [1.0],
                    'class_weight': ['balanced'],
                    'max_iter': [10000]
                }
            ),
            "KNN": (
                KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=self.n_jobs
                ),
                {
                    'n_neighbors': [3], #, 5, 7],
                    'weights': ['uniform'], #, 'distance'],
                    'metric': ['euclidean'] #, 'manhattan']
                }
            ),
            "MLP": (
                MLPClassifier(
                    hidden_layer_sizes=(128, 64), 
                    max_iter=400, 
                    random_state=self.random_state
                ),
                {
                    'hidden_layer_sizes': [(128, 64)],
                    'learning_rate_init': [0.01],
                    'max_iter': [50]
                }
            )
        }
        return configs
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance and print metrics.
        """
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Handle probability predictions (not all models support predict_proba)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except AttributeError:
            # For models like LinearSVC that don't have predict_proba
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred_proba = None
                roc_auc = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        
        # Print metrics
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store metrics for CSV export
        metrics_dict = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc if roc_auc is not None else 'N/A'
        }
        self.evaluation_results.append(metrics_dict)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
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
    
    def plot_decision_tree(self, model, feature_names, model_name, output_path, max_depth=3):
        """
        Plot and save decision tree visualization.
        
        Args:
            model: Trained DecisionTreeClassifier
            feature_names: List of feature names
            model_name: Name of the model
            output_path: Path to save the tree plot
            max_depth: Maximum depth to display (to avoid overcrowding)
        """
        plt.figure(figsize=(20, 12))
        plot_tree(model, 
                  feature_names=feature_names,
                  class_names=['Not Successful', 'Successful'],
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  max_depth=max_depth)
        plt.title(f'Decision Tree Visualization - {model_name} (max_depth={max_depth} shown)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Decision tree visualization saved: {output_path}")
    
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
        Save trained models to disk.
        Note: Scaler is saved in feature_engineering.py
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {model_path}")
    
    def save_evaluation_metrics(self, output_dir='models'):
        """
        Save evaluation metrics to CSV file.
        """
        if not self.evaluation_results:
            print("No evaluation results to save.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
        
        metrics_df = pd.DataFrame(self.evaluation_results)
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"\nEvaluation metrics saved: {metrics_path}")
        print("\nMetrics Summary:")
        print(metrics_df.to_string(index=False))
        
        # Save best hyperparameters to separate file
        if self.best_params:
            params_path = os.path.join(output_dir, 'best_hyperparameters.txt')
            with open(params_path, 'w') as f:
                f.write("Best Hyperparameters from GridSearchCV\n")
                f.write("="*70 + "\n\n")
                for model_name, params in self.best_params.items():
                    f.write(f"{model_name}:\n")
                    for param, value in params.items():
                        f.write(f"  {param}: {value}\n")
                    f.write("\n")
            print(f"Best hyperparameters saved: {params_path}")
    
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
        
        # Get all model configurations
        model_configs = self.get_model_configs()
        
        # Create output directories
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Train and evaluate all models
        all_results = {}
        for model_name, (base_model, param_grid) in model_configs.items():
            print("\n" + "="*70)
            print(f"TRAINING {model_name.upper()} WITH CROSS-VALIDATION")
            print("="*70)
            
            # Perform GridSearchCV for hyperparameter tuning
            if param_grid:
                print(f"\nPerforming GridSearchCV with {len(param_grid)} hyperparameters...")
                print(f"Parameter grid: {param_grid}")
                
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='precision',
                    n_jobs=self.n_jobs,
                    verbose=1
                )
                
                start_time = time.time()
                grid_search.fit(X_train, y_train)
                duration = time.time() - start_time
                
                print(f"\nGridSearchCV completed in {duration:.2f} seconds")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
                
                # Use best estimator
                model = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
            else:
                # Train without hyperparameter tuning
                start_time = time.time()
                base_model.fit(X_train, y_train)
                duration = time.time() - start_time
                print(f"{model_name} training completed in {duration:.2f} seconds")
                model = base_model
            
            self.models[model_name] = model
            
            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test, model_name)
            all_results[model_name] = results
            
            # Plot confusion matrix for each model
            self.plot_confusion_matrix(
                results['confusion_matrix'], model_name,
                output_path=os.path.join(plots_dir, f'{model_name.lower()}_confusion_matrix.png')
            )
            
            # Plot feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(
                    model, feature_names, model_name,
                    output_path=os.path.join(plots_dir, f'{model_name.lower()}_feature_importance.png')
                )
            
            # Plot decision tree for DecisionTree model
            if model_name == "DecisionTree":
                self.plot_decision_tree(
                    model, feature_names, model_name,
                    output_path=os.path.join(plots_dir, f'{model_name.lower()}_tree.png'),
                    max_depth=5  # Show only top 5 levels for readability
                )
        
        # Save models
        self.save_models(output_dir)
        
        # Save evaluation metrics to CSV
        self.save_evaluation_metrics(output_dir)
        
        print("\n" + "="*70)
        print("MODEL TRAINING PIPELINE COMPLETED")
        print("="*70)
        
        return {
            'models': self.models,
            'all_results': all_results,
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
