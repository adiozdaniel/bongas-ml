"""
Data loader for BONGAS-ML

Handles loading training data from PostgreSQL database with proper
feature engineering, preprocessing, and data validation for recommendation models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger

from ..utils.logging import setup_logging


class TrainingDataLoader:
    """Data loader for training recommendation models"""
    
    def __init__(self, db_url: str):
        """
        Initialize data loader
        
        Args:
            db_url: PostgreSQL database connection URL
        """
        self.db_url = db_url
        self.connection = None
        
        logger.info(f"Data loader initialized with database: {db_url}")
    
    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def load_customer_data(
        self,
        customer_id: str,
        days_back: int = 30,
        min_interactions: int = 10
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load training data for a specific customer
        
        Args:
            customer_id: Customer identifier
            days_back: Number of days of data to load
            min_interactions: Minimum number of interactions required
        
        Returns:
            Dictionary containing training tensors or None if insufficient data
        """
        try:
            self.connect()
            
            # Load interaction data
            interactions_df = self._load_interactions(customer_id, days_back)
            
            if len(interactions_df) < min_interactions:
                logger.warning(f"Insufficient data for customer {customer_id}: {len(interactions_df)} interactions")
                return None
            
            # Load user and item metadata
            user_features_df = self._load_user_features(customer_id, interactions_df)
            item_features_df = self._load_item_features(interactions_df)
            
            # Preprocess and create tensors
            training_data = self._create_training_tensors(
                interactions_df, user_features_df, item_features_df
            )
            
            logger.info(f"Loaded training data for customer {customer_id}:")
            logger.info(f"  Interactions: {len(interactions_df)}")
            logger.info(f"  Users: {len(user_features_df)}")
            logger.info(f"  Items: {len(item_features_df)}")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to load data for customer {customer_id}: {e}")
            return None
        finally:
            self.disconnect()
    
    def load_all_customers(self) -> List[str]:
        """Get list of all customers in the database"""
        try:
            self.connect()
            
            query = """
                SELECT DISTINCT customer_id 
                FROM interactions 
                ORDER BY customer_id
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
            
            customers = [row['customer_id'] for row in results]
            logger.info(f"Found {len(customers)} customers in database")
            
            return customers
            
        except Exception as e:
            logger.error(f"Failed to load customer list: {e}")
            return []
        finally:
            self.disconnect()
    
    def _load_interactions(
        self,
        customer_id: str,
        days_back: int
    ) -> pd.DataFrame:
        """Load interaction data for customer"""
        
        query = """
            SELECT 
                user_id,
                item_id,
                interaction_type,
                interaction_value,
                timestamp
            FROM interactions 
            WHERE customer_id = %s 
                AND timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (customer_id, days_back))
            results = cursor.fetchall()
        
        df = pd.DataFrame([dict(row) for row in results])
        
        if len(df) == 0:
            logger.warning(f"No interactions found for customer {customer_id}")
        
        return df
    
    def _load_user_features(
        self,
        customer_id: str,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Load user features"""
        
        user_ids = interactions_df['user_id'].unique().tolist()
        
        query = """
            SELECT 
                user_id,
                age,
                gender,
                country,
                device_type,
                language,
                created_at
            FROM user_profiles 
            WHERE customer_id = %s 
                AND user_id = ANY(%s)
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (customer_id, user_ids))
            results = cursor.fetchall()
        
        df = pd.DataFrame([dict(row) for row in results])
        
        if len(df) == 0:
            logger.warning(f"No user features found for customer {customer_id}")
        
        return df
    
    def _load_item_features(
        self,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Load item features"""
        
        item_ids = interactions_df['item_id'].unique().tolist()
        
        query = """
            SELECT 
                item_id,
                category,
                subcategory,
                price,
                popularity_score,
                created_at
            FROM item_metadata 
            WHERE item_id = ANY(%s)
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (item_ids,))
            results = cursor.fetchall()
        
        df = pd.DataFrame([dict(row) for row in results])
        
        if len(df) == 0:
            logger.warning(f"No item features found")
        
        return df
    
    def _create_training_tensors(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Create training tensors from dataframes"""
        
        # Merge data
        merged_df = interactions_df.merge(
            user_features_df, on='user_id', how='left'
        ).merge(
            item_features_df, on='item_id', how='left'
        )
        
        # Handle missing values
        merged_df = self._handle_missing_values(merged_df)
        
        # Create features
        user_features = self._create_user_features(merged_df, user_features_df)
        item_features = self._create_item_features(merged_df, item_features_df)
        
        # Create targets (convert interaction values to binary or continuous)
        targets = self._create_targets(merged_df)
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'targets': targets
        }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _create_user_features(
        self,
        merged_df: pd.DataFrame,
        user_features_df: pd.DataFrame
    ) -> torch.Tensor:
        """Create user feature tensors"""
        
        # Create feature matrix
        features = []
        
        # Age (normalized)
        if 'age' in merged_df.columns:
            age_normalized = (merged_df['age'] - merged_df['age'].min()) / (merged_df['age'].max() - merged_df['age'].min())
            features.append(age_normalized.values.reshape(-1, 1))
        
        # Gender (one-hot)
        if 'gender' in merged_df.columns:
            gender_dummies = pd.get_dummies(merged_df['gender'], prefix='gender')
            features.append(gender_dummies.values)
        
        # Country (one-hot)
        if 'country' in merged_df.columns:
            country_dummies = pd.get_dummies(merged_df['country'], prefix='country')
            features.append(country_dummies.values)
        
        # Device type (one-hot)
        if 'device_type' in merged_df.columns:
            device_dummies = pd.get_dummies(merged_df['device_type'], prefix='device')
            features.append(device_dummies.values)
        
        # Language (one-hot)
        if 'language' in merged_df.columns:
            language_dummies = pd.get_dummies(merged_df['language'], prefix='language')
            features.append(language_dummies.values)
        
        if len(features) == 0:
            # Fallback to zeros if no features available
            num_samples = len(merged_df)
            features = [torch.zeros(num_samples, 1)]
        
        # Concatenate all features
        user_features = torch.cat([torch.tensor(f, dtype=torch.float32) for f in features], dim=1)
        
        return user_features
    
    def _create_item_features(
        self,
        merged_df: pd.DataFrame,
        item_features_df: pd.DataFrame
    ) -> torch.Tensor:
        """Create item feature tensors"""
        
        features = []
        
        # Category (one-hot)
        if 'category' in merged_df.columns:
            category_dummies = pd.get_dummies(merged_df['category'], prefix='category')
            features.append(category_dummies.values)
        
        # Subcategory (one-hot)
        if 'subcategory' in merged_df.columns:
            subcategory_dummies = pd.get_dummies(merged_df['subcategory'], prefix='subcategory')
            features.append(subcategory_dummies.values)
        
        # Price (normalized)
        if 'price' in merged_df.columns:
            price_normalized = (merged_df['price'] - merged_df['price'].min()) / (merged_df['price'].max() - merged_df['price'].min())
            features.append(price_normalized.values.reshape(-1, 1))
        
        # Popularity score (normalized)
        if 'popularity_score' in merged_df.columns:
            popularity_normalized = (merged_df['popularity_score'] - merged_df['popularity_score'].min()) / (merged_df['popularity_score'].max() - merged_df['popularity_score'].min())
            features.append(popularity_normalized.values.reshape(-1, 1))
        
        if len(features) == 0:
            # Fallback to zeros if no features available
            num_samples = len(merged_df)
            features = [torch.zeros(num_samples, 1)]
        
        # Concatenate all features
        item_features = torch.cat([torch.tensor(f, dtype=torch.float32) for f in features], dim=1)
        
        return item_features
    
    def _create_targets(self, merged_df: pd.DataFrame) -> torch.Tensor:
        """Create target tensors"""
        
        if 'interaction_value' in merged_df.columns:
            # Use interaction values directly (for regression)
            targets = torch.tensor(merged_df['interaction_value'].values, dtype=torch.float32)
        else:
            # Binary targets (for classification)
            targets = torch.ones(len(merged_df), dtype=torch.float32)
        
        return targets
    
    def validate_data(
        self,
        training_data: Dict[str, torch.Tensor],
        customer_id: str
    ) -> bool:
        """Validate training data quality"""
        
        try:
            user_features = training_data['user_features']
            item_features = training_data['item_features']
            targets = training_data['targets']
            
            # Check shapes
            num_samples = len(targets)
            if user_features.shape[0] != num_samples or item_features.shape[0] != num_samples:
                logger.error(f"Shape mismatch for customer {customer_id}")
                return False
            
            # Check for NaN values
            if torch.isnan(user_features).any() or torch.isnan(item_features).any() or torch.isnan(targets).any():
                logger.error(f"NaN values found in data for customer {customer_id}")
                return False
            
            # Check target distribution
            positive_ratio = (targets > 0).float().mean()
            if positive_ratio < 0.01 or positive_ratio > 0.99:
                logger.warning(f"Extreme class imbalance for customer {customer_id}: {positive_ratio:.3f}")
            
            logger.info(f"Data validation passed for customer {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed for customer {customer_id}: {e}")
            return False
    
    def get_data_statistics(
        self,
        training_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Get data statistics"""
        
        user_features = training_data['user_features']
        item_features = training_data['item_features']
        targets = training_data['targets']
        
        stats = {
            'num_samples': len(targets),
            'user_feature_dim': user_features.shape[1],
            'item_feature_dim': item_features.shape[1],
            'target_mean': targets.mean().item(),
            'target_std': targets.std().item(),
            'positive_ratio': (targets > 0).float().mean().item(),
            'user_feature_mean': user_features.mean().item(),
            'user_feature_std': user_features.std().item(),
            'item_feature_mean': item_features.mean().item(),
            'item_feature_std': item_features.std().item(),
        }
        
        return stats