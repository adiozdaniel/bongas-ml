"""
Feature transformers for BONGAS-ML

Provides advanced feature transformation and preprocessing including:
- Feature scaling and normalization
- Categorical encoding
- Feature selection
- Dimensionality reduction
- Custom feature transformations
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)


class FeatureTransformer:
    """Advanced feature transformation and preprocessing"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}
        self.pca_models = {}
        self.polynomial_features = {}
        self.feature_names = {}
        self.transform_config = {}
        
        logger.info("Feature transformer initialized")
    
    def configure_transformations(
        self,
        config: Dict[str, Any]
    ) -> None:
        """
        Configure transformation pipeline
        
        Args:
            config: Transformation configuration
        """
        
        self.transform_config = config
        logger.info("Feature transformation configuration updated")
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_types: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit and apply transformations
        
        Args:
            data: Input data
            target: Target variable (for supervised transformations)
            feature_types: Dictionary of feature type mappings
        
        Returns:
            Tuple of (transformed_data, transformation_metadata)
        """
        
        try:
            transformed_data = data.copy()
            metadata = {}
            
            # Get feature types if not provided
            if feature_types is None:
                feature_types = self._infer_feature_types(data)
            
            # Apply transformations based on configuration
            if 'scaling' in self.transform_config:
                transformed_data = self._apply_scaling(transformed_data, feature_types)
                metadata['scaling'] = self._get_scaling_metadata()
            
            if 'encoding' in self.transform_config:
                transformed_data = self._apply_encoding(transformed_data, feature_types)
                metadata['encoding'] = self._get_encoding_metadata()
            
            if 'polynomial' in self.transform_config:
                transformed_data = self._apply_polynomial_features(transformed_data, feature_types)
                metadata['polynomial'] = self._get_polynomial_metadata()
            
            if 'selection' in self.transform_config and target is not None:
                transformed_data = self._apply_feature_selection(
                    transformed_data, target, feature_types
                )
                metadata['selection'] = self._get_selection_metadata()
            
            if 'pca' in self.transform_config:
                transformed_data = self._apply_pca(transformed_data, feature_types)
                metadata['pca'] = self._get_pca_metadata()
            
            # Store feature names
            self.feature_names = transformed_data.columns.tolist()
            
            logger.info(f"Feature transformation completed: {len(transformed_data.columns)} features")
            return transformed_data, metadata
            
        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data
        
        Args:
            data: New data to transform
        
        Returns:
            Transformed DataFrame
        """
        
        try:
            transformed_data = data.copy()
            
            # Apply transformations in order
            if 'scaling' in self.transform_config:
                transformed_data = self._transform_scaling(transformed_data)
            
            if 'encoding' in self.transform_config:
                transformed_data = self._transform_encoding(transformed_data)
            
            if 'polynomial' in self.transform_config:
                transformed_data = self._transform_polynomial(transformed_data)
            
            if 'selection' in self.transform_config:
                transformed_data = self._transform_selection(transformed_data)
            
            if 'pca' in self.transform_config:
                transformed_data = self._transform_pca(transformed_data)
            
            logger.info("New data transformation completed")
            return transformed_data
            
        except Exception as e:
            logger.error(f"New data transformation failed: {e}")
            raise
    
    def _infer_feature_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Infer feature types from data"""
        
        categorical = []
        numerical = []
        
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                categorical.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                numerical.append(col)
        
        return {'categorical': categorical, 'numerical': numerical}
    
    def _apply_scaling(
        self,
        data: pd.DataFrame,
        feature_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply feature scaling"""
        
        scaling_config = self.transform_config.get('scaling', {})
        scaler_type = scaling_config.get('type', 'standard')
        columns = scaling_config.get('columns', feature_types['numerical'])
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        if columns:
            scaled_data = scaler.fit_transform(data[columns])
            scaled_df = pd.DataFrame(scaled_data, columns=columns, index=data.index)
            data = data.drop(columns=columns)
            data = pd.concat([data, scaled_df], axis=1)
            self.scalers['numerical'] = scaler
        
        return data
    
    def _apply_encoding(
        self,
        data: pd.DataFrame,
        feature_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply categorical encoding"""
        
        encoding_config = self.transform_config.get('encoding', {})
        encoding_type = encoding_config.get('type', 'onehot')
        columns = encoding_config.get('columns', feature_types['categorical'])
        
        if encoding_type == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            if columns:
                encoded_data = encoder.fit_transform(data[columns])
                feature_names = encoder.get_feature_names_out(columns)
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, encoded_df], axis=1)
                self.encoders['categorical'] = encoder
        
        elif encoding_type == 'label':
            for col in columns:
                encoder = LabelEncoder()
                data[f'{col}_encoded'] = encoder.fit_transform(data[col].astype(str))
                self.encoders[col] = encoder
        
        return data
    
    def _apply_polynomial_features(
        self,
        data: pd.DataFrame,
        feature_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply polynomial feature expansion"""
        
        poly_config = self.transform_config.get('polynomial', {})
        degree = poly_config.get('degree', 2)
        columns = poly_config.get('columns', feature_types['numerical'])
        
        if columns and degree > 1:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = poly.fit_transform(data[columns])
            feature_names = poly.get_feature_names_out(columns)
            poly_df = pd.DataFrame(poly_data, columns=feature_names, index=data.index)
            data = data.drop(columns=columns)
            data = pd.concat([data, poly_df], axis=1)
            self.polynomial_features['numerical'] = poly
        
        return data
    
    def _apply_feature_selection(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        feature_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply feature selection"""
        
        selection_config = self.transform_config.get('selection', {})
        method = selection_config.get('method', 'f_classif' if target.dtype == 'object' else 'f_regression')
        k = selection_config.get('k', 10)
        
        # Get numerical features for selection
        numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        if len(numerical_cols) > k:
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=k)
            elif method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                raise ValueError(f"Unknown selection method: {method}")
            
            selected_data = selector.fit_transform(data[numerical_cols], target)
            selected_features = [numerical_cols[i] for i in selector.get_support(indices=True)]
            
            # Create new dataframe with selected features
            selected_df = pd.DataFrame(selected_data, columns=selected_features, index=data.index)
            
            # Keep non-numerical features
            other_cols = [col for col in data.columns if col not in numerical_cols]
            if other_cols:
                other_df = data[other_cols]
                data = pd.concat([other_df, selected_df], axis=1)
            else:
                data = selected_df
            
            self.selectors['numerical'] = selector
        
        return data
    
    def _apply_pca(
        self,
        data: pd.DataFrame,
        feature_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply PCA dimensionality reduction"""
        
        pca_config = self.transform_config.get('pca', {})
        n_components = pca_config.get('n_components', 0.95)  # Keep 95% of variance
        columns = pca_config.get('columns', feature_types['numerical'])
        
        if columns:
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(data[columns])
            
            # Create new feature names
            pca_columns = [f'pca_{i}' for i in range(pca_data.shape[1])]
            pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
            
            # Remove original columns and add PCA components
            data = data.drop(columns=columns)
            data = pd.concat([data, pca_df], axis=1)
            
            self.pca_models['numerical'] = pca
        
        return data
    
    def _transform_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling to new data"""
        
        if 'numerical' in self.scalers:
            scaler = self.scalers['numerical']
            columns = [col for col in data.columns if col in scaler.feature_names_in_]
            if columns:
                scaled_data = scaler.transform(data[columns])
                scaled_df = pd.DataFrame(scaled_data, columns=columns, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, scaled_df], axis=1)
        
        return data
    
    def _transform_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoding to new data"""
        
        if 'categorical' in self.encoders:
            encoder = self.encoders['categorical']
            columns = [col for col in data.columns if col in encoder.feature_names_in_]
            if columns:
                encoded_data = encoder.transform(data[columns])
                feature_names = encoder.get_feature_names_out(columns)
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, encoded_df], axis=1)
        
        return data
    
    def _transform_polynomial(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted polynomial features to new data"""
        
        if 'numerical' in self.polynomial_features:
            poly = self.polynomial_features['numerical']
            columns = [col for col in data.columns if col in poly.feature_names_in_]
            if columns:
                poly_data = poly.transform(data[columns])
                feature_names = poly.get_feature_names_out(columns)
                poly_df = pd.DataFrame(poly_data, columns=feature_names, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, poly_df], axis=1)
        
        return data
    
    def _transform_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted feature selection to new data"""
        
        if 'numerical' in self.selectors:
            selector = self.selectors['numerical']
            columns = [col for col in data.columns if col in selector.feature_names_in_]
            if columns:
                selected_data = selector.transform(data[columns])
                selected_features = [columns[i] for i in selector.get_support(indices=True)]
                selected_df = pd.DataFrame(selected_data, columns=selected_features, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, selected_df], axis=1)
        
        return data
    
    def _transform_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted PCA to new data"""
        
        if 'numerical' in self.pca_models:
            pca = self.pca_models['numerical']
            columns = [col for col in data.columns if col in pca.feature_names_in_]
            if columns:
                pca_data = pca.transform(data[columns])
                pca_columns = [f'pca_{i}' for i in range(pca_data.shape[1])]
                pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
                data = data.drop(columns=columns)
                data = pd.concat([data, pca_df], axis=1)
        
        return data
    
    def _get_scaling_metadata(self) -> Dict[str, Any]:
        """Get scaling transformation metadata"""
        
        metadata = {}
        if 'numerical' in self.scalers:
            scaler = self.scalers['numerical']
            metadata = {
                'type': scaler.__class__.__name__,
                'feature_names': scaler.feature_names_in_.tolist(),
                'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
            }
        
        return metadata
    
    def _get_encoding_metadata(self) -> Dict[str, Any]:
        """Get encoding transformation metadata"""
        
        metadata = {}
        if 'categorical' in self.encoders:
            encoder = self.encoders['categorical']
            metadata = {
                'type': encoder.__class__.__name__,
                'feature_names': encoder.feature_names_in_.tolist(),
                'categories': [cat.tolist() for cat in encoder.categories_] if hasattr(encoder, 'categories_') else None
            }
        
        return metadata
    
    def _get_polynomial_metadata(self) -> Dict[str, Any]:
        """Get polynomial transformation metadata"""
        
        metadata = {}
        if 'numerical' in self.polynomial_features:
            poly = self.polynomial_features['numerical']
            metadata = {
                'degree': poly.degree,
                'include_bias': poly.include_bias,
                'feature_names': poly.feature_names_in_.tolist(),
                'output_features': poly.get_feature_names_out().tolist()
            }
        
        return metadata
    
    def _get_selection_metadata(self) -> Dict[str, Any]:
        """Get feature selection metadata"""
        
        metadata = {}
        if 'numerical' in self.selectors:
            selector = self.selectors['numerical']
            metadata = {
                'k': selector.k,
                'scores': selector.scores_.tolist(),
                'pvalues': selector.pvalues_.tolist(),
                'selected_features': selector.get_feature_names_out().tolist()
            }
        
        return metadata
    
    def _get_pca_metadata(self) -> Dict[str, Any]:
        """Get PCA transformation metadata"""
        
        metadata = {}
        if 'numerical' in self.pca_models:
            pca = self.pca_models['numerical']
            metadata = {
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'feature_names': pca.feature_names_in_.tolist()
            }
        
        return metadata
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """Get feature importance from transformations"""
        
        importance = {}
        
        if 'selection' in self.selectors:
            selector = self.selectors['selection']
            importance['selection'] = selector.scores_.tolist()
        
        if 'pca' in self.pca_models:
            pca = self.pca_models['pca']
            importance['pca_variance'] = pca.explained_variance_ratio_.tolist()
        
        return importance
    
    def reset(self) -> None:
        """Reset all fitted transformers"""
        
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}
        self.pca_models = {}
        self.polynomial_features = {}
        self.feature_names = {}
        self.transform_config = {}
        
        logger.info("Feature transformer reset")