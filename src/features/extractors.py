"""
Feature extractors for BONGAS-ML

Provides automatic feature extraction from raw data including:
- Categorical feature encoding
- Numerical feature scaling
- Temporal feature processing
- Text feature extraction
- Feature validation and cleaning
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureExtractor:
    """Extracts and processes features from raw data"""
    
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_stats = {}
        self.feature_types = {}
        
        logger.info("Feature extractor initialized")
    
    def extract_features(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract features from raw data
        
        Args:
            data: Input data (DataFrame or dict)
            target_column: Name of target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
        
        Returns:
            Tuple of (processed_data, feature_metadata)
        """
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            
            # Auto-detect column types if not provided
            if categorical_columns is None or numerical_columns is None:
                categorical_columns, numerical_columns = self._detect_column_types(
                    data, target_column
                )
            
            # Store feature types
            self.feature_types = {
                'categorical': categorical_columns,
                'numerical': numerical_columns,
                'target': target_column
            }
            
            # Process features
            processed_data = data.copy()
            
            # Handle categorical features
            for col in categorical_columns:
                if col in processed_data.columns:
                    processed_data = self._process_categorical_column(processed_data, col)
            
            # Handle numerical features
            for col in numerical_columns:
                if col in processed_data.columns:
                    processed_data = self._process_numerical_column(processed_data, col)
            
            # Handle target column
            if target_column and target_column in processed_data.columns:
                processed_data = self._process_target_column(processed_data, target_column)
            
            # Generate feature metadata
            feature_metadata = self._generate_feature_metadata(processed_data)
            
            logger.info(f"Feature extraction completed: {len(processed_data.columns)} features")
            return processed_data, feature_metadata
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _detect_column_types(
        self,
        data: pd.DataFrame,
        target_column: Optional[str]
    ) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and numerical columns"""
        
        categorical_columns = []
        numerical_columns = []
        
        for col in data.columns:
            if col == target_column:
                continue
                
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                numerical_columns.append(col)
            else:
                # Try to infer type
                try:
                    pd.to_numeric(data[col])
                    numerical_columns.append(col)
                except (ValueError, TypeError):
                    categorical_columns.append(col)
        
        logger.info(f"Auto-detected {len(categorical_columns)} categorical and {len(numerical_columns)} numerical features")
        return categorical_columns, numerical_columns
    
    def _process_categorical_column(
        self,
        data: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Process categorical column with encoding"""
        
        try:
            # Handle missing values
            data[column] = data[column].fillna('unknown')
            
            # Create encoder if not exists
            if column not in self.categorical_encoders:
                encoder = LabelEncoder()
                # Fit on non-null values
                valid_values = data[column].dropna().unique()
                encoder.fit(valid_values)
                self.categorical_encoders[column] = encoder
            
            # Transform values
            encoder = self.categorical_encoders[column]
            try:
                data[f'{column}_encoded'] = encoder.transform(data[column])
            except ValueError:
                # Handle unseen categories
                data[f'{column}_encoded'] = data[column].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            
            # Store statistics
            self.feature_stats[column] = {
                'type': 'categorical',
                'unique_values': len(encoder.classes_),
                'missing_count': data[column].isnull().sum(),
                'encoder_classes': encoder.classes_.tolist()
            }
            
            logger.debug(f"Processed categorical column: {column}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to process categorical column {column}: {e}")
            raise
    
    def _process_numerical_column(
        self,
        data: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Process numerical column with scaling"""
        
        try:
            # Handle missing values
            data[column] = data[column].fillna(data[column].median())
            
            # Create scaler if not exists
            if column not in self.numerical_scalers:
                scaler = StandardScaler()
                scaler.fit(data[[column]])
                self.numerical_scalers[column] = scaler
            
            # Transform values
            scaler = self.numerical_scalers[column]
            data[f'{column}_scaled'] = scaler.transform(data[[column]]).flatten()
            
            # Store statistics
            self.feature_stats[column] = {
                'type': 'numerical',
                'mean': float(data[column].mean()),
                'std': float(data[column].std()),
                'min': float(data[column].min()),
                'max': float(data[column].max()),
                'missing_count': data[column].isnull().sum()
            }
            
            logger.debug(f"Processed numerical column: {column}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to process numerical column {column}: {e}")
            raise
    
    def _process_target_column(
        self,
        data: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Process target column"""
        
        try:
            # Handle missing values
            data[column] = data[column].fillna(data[column].median())
            
            # Store target statistics
            self.feature_stats[column] = {
                'type': 'target',
                'mean': float(data[column].mean()),
                'std': float(data[column].std()),
                'min': float(data[column].min()),
                'max': float(data[column].max()),
                'missing_count': data[column].isnull().sum()
            }
            
            logger.debug(f"Processed target column: {column}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to process target column {column}: {e}")
            raise
    
    def _generate_feature_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata for processed features"""
        
        metadata = {
            'feature_types': self.feature_types,
            'feature_stats': self.feature_stats,
            'processed_columns': data.columns.tolist(),
            'total_features': len(data.columns),
            'extraction_timestamp': str(pd.Timestamp.now())
        }
        
        return metadata
    
    def transform_new_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Transform new data using fitted extractors
        
        Args:
            data: New data to transform
        
        Returns:
            Transformed DataFrame
        """
        
        try:
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            
            processed_data = data.copy()
            
            # Apply categorical transformations
            for col, encoder in self.categorical_encoders.items():
                if col in processed_data.columns:
                    processed_data = self._process_categorical_column(processed_data, col)
            
            # Apply numerical transformations
            for col, scaler in self.numerical_scalers.items():
                if col in processed_data.columns:
                    processed_data = self._process_numerical_column(processed_data, col)
            
            logger.info("New data transformation completed")
            return processed_data
            
        except Exception as e:
            logger.error(f"New data transformation failed: {e}")
            raise
    
    def get_feature_importance(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        """
        Calculate basic feature importance
        
        Args:
            data: Processed data
            target_column: Target column name
        
        Returns:
            Dictionary of feature importances
        """
        
        try:
            importances = {}
            
            for col in data.columns:
                if col == target_column:
                    continue
                
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Calculate correlation with target
                    correlation = abs(data[col].corr(data[target_column]))
                    importances[col] = float(correlation)
                else:
                    # For categorical features, use mutual information concept
                    # This is a simplified version
                    unique_values = data[col].nunique()
                    importances[col] = 1.0 / (unique_values + 1)
            
            # Sort by importance
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            
            logger.info("Feature importance calculation completed")
            return importances
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def validate_features(
        self,
        data: pd.DataFrame,
        min_missing_ratio: float = 0.5,
        min_unique_ratio: float = 0.01
    ) -> Dict[str, List[str]]:
        """
        Validate features for quality issues
        
        Args:
            data: Data to validate
            min_missing_ratio: Minimum ratio of missing values to flag
            min_unique_ratio: Minimum ratio of unique values to flag
        
        Returns:
            Dictionary of validation issues
        """
        
        issues = {
            'high_missing': [],
            'low_unique': [],
            'constant': [],
            'all_missing': []
        }
        
        for col in data.columns:
            # Check missing values
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > min_missing_ratio:
                issues['high_missing'].append(col)
            
            if missing_ratio == 1.0:
                issues['all_missing'].append(col)
            
            # Check unique values
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < min_unique_ratio and unique_ratio > 0:
                issues['low_unique'].append(col)
            
            # Check constant values
            if data[col].nunique() == 1:
                issues['constant'].append(col)
        
        logger.info(f"Feature validation completed: {sum(len(v) for v in issues.values())} issues found")
        return issues
    
    def get_categorical_mapping(self, column: str) -> Optional[Dict[int, str]]:
        """Get mapping from encoded values to original categories"""
        
        if column in self.categorical_encoders:
            encoder = self.categorical_encoders[column]
            return {i: label for i, label in enumerate(encoder.classes_)}
        return None
    
    def get_numerical_stats(self, column: str) -> Optional[Dict[str, float]]:
        """Get statistics for numerical column"""
        
        if column in self.feature_stats and self.feature_stats[column]['type'] == 'numerical':
            return self.feature_stats[column]
        return None
    
    def reset(self) -> None:
        """Reset all fitted extractors"""
        
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_stats = {}
        self.feature_types = {}
        
        logger.info("Feature extractor reset")