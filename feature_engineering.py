import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering and model training for YouTube trend prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def engineer_features(self, df):
        """
        Create features from video metadata and channel characteristics.
        """
        features_df = df.copy()
        
        # === VIDEO METADATA FEATURES ===
        
        # Title features
        features_df['title_length'] = features_df['title'].fillna('').str.len()
        features_df['title_word_count'] = features_df['title'].fillna('').str.split().str.len()
        features_df['title_has_question'] = features_df['title'].fillna('').str.contains(r'\?').astype(int)
        features_df['title_has_exclamation'] = features_df['title'].fillna('').str.contains('!').astype(int)
        features_df['title_uppercase_ratio'] = features_df['title'].fillna('').apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Description features
        features_df['description_length'] = features_df['description'].fillna('').str.len()
        features_df['description_word_count'] = features_df['description'].fillna('').str.split().str.len()
        features_df['has_description'] = (features_df['description_length'] > 0).astype(int)
        
        # Tags features
        features_df['num_tags'] = features_df['tags'].fillna('').apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Video duration (convert to minutes)
        features_df['duration_minutes'] = features_df['duration'].fillna(0) / 60
        features_df['is_short_video'] = (features_df['duration_minutes'] < 5).astype(int)
        features_df['is_long_video'] = (features_df['duration_minutes'] > 20).astype(int)
        
        # Upload timing features
        features_df['upload_date_dt'] = pd.to_datetime(features_df['upload_date'])
        features_df['upload_day_of_week'] = features_df['upload_date_dt'].dt.dayofweek
        features_df['upload_hour'] = features_df['upload_date_dt'].dt.hour
        features_df['is_weekend_upload'] = (features_df['upload_day_of_week'] >= 5).astype(int)
        
        # === CHANNEL CHARACTERISTICS ===
        
        # Channel size and popularity
        features_df['channel_subscribers'] = features_df['subscribers_cc'].fillna(0)
        features_df['channel_total_videos'] = features_df['videos_cc'].fillna(0)
        features_df['channel_subscriber_rank'] = features_df['subscriber_rank_sb'].fillna(
            features_df['subscriber_rank_sb'].max()
        )
        
        # Channel engagement metrics
        features_df['avg_views_per_video'] = features_df['channel_subscribers'] / (
            features_df['channel_total_videos'] + 1
        )
        features_df['subscriber_to_video_ratio'] = features_df['channel_subscribers'] / (
            features_df['channel_total_videos'] + 1
        )
        
        # === EARLY ENGAGEMENT INDICATORS ===
        features_df['num_comments_7d'] = features_df['num_comms'].fillna(0)
        features_df['comments_per_view'] = features_df['num_comments_7d'] / (
            features_df['views_at_7d'] + 1
        )
        
        # Category encoding
        if 'categories' in features_df.columns:
            features_df['category'] = features_df['categories'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
            )
            
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                features_df['category_encoded'] = self.label_encoders['category'].fit_transform(
                    features_df['category'].fillna('Unknown')
                )
            else:
                features_df['category_encoded'] = self.label_encoders['category'].transform(
                    features_df['category'].fillna('Unknown')
                )
        
        return features_df
    
    def prepare_features_and_target(self, df):
        """
        Select final features and prepare for modeling.
        """
        feature_columns = [
            # Title features
            'title_length', 'title_word_count', 'title_has_question', 
            'title_has_exclamation', 'title_uppercase_ratio',
            
            # Description features
            'description_length', 'description_word_count', 'has_description',
            
            # Tags and content
            'num_tags', 'duration_minutes', 'is_short_video', 'is_long_video',
            
            # Upload timing
            'upload_day_of_week', 'upload_hour', 'is_weekend_upload',
            
            # Channel characteristics
            'channel_subscribers', 'channel_total_videos', 'channel_subscriber_rank',
            'avg_views_per_video', 'subscriber_to_video_ratio',
            
            # Early engagement
            'num_comments_7d', 'comments_per_view',
            
            # Category
            'category_encoded'
        ]
        
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].fillna(0)
        y = df['is_successful']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Features used: {feature_columns}")
        
        return X, y, feature_columns
    
    def train_model(self, X, y):
        """
        Train Random Forest and Gradient Boosting classifiers.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n" + "="*50)
        print("Training Random Forest Classifier...")
        print("="*50)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_rf = rf_model.predict(X_test_scaled)
        y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nRandom Forest Results:")
        print(classification_report(y_test, y_pred_rf))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
        
        print("\n" + "="*50)
        print("Training Gradient Boosting Classifier...")
        print("="*50)
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_gb = gb_model.predict(X_test_scaled)
        y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nGradient Boosting Results:")
        print(classification_report(y_test, y_pred_gb))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
        
        return rf_model, gb_model, X_test_scaled, y_test
    
    def plot_feature_importance(self, model, feature_names, model_name="Model"):
        """
        Plot feature importance.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
