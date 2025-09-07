import pandas as pd
import numpy as np


class BigMartPreprocessor:
    """
    Advanced preprocessing pipeline for BigMart sales data
    Fixed version to handle NaN values properly
    """
    def __init__(self, pprint=False):
        self.item_stats = {}
        self.outlet_stats = {}
        self.target_encoders = {}
        self.categorical_columns = []
        self.dummy_columns = {}  # Store expected dummy columns
        self.is_fitted = False
        import pandas as pd
        import numpy as np


        class BigMartPreprocessor:
            """
            Advanced preprocessing pipeline for BigMart sales data
            Fixed version to handle NaN values properly
            """
            def __init__(self, pprint=False):
                self.item_stats = {}
                self.outlet_stats = {}
                self.target_encoders = {}
                self.categorical_columns = []
                self.dummy_columns = {}  # Store expected dummy columns
                self.is_fitted = False
        
            def fit(self, X, y=None):
                """Fit the preprocessor on training data"""
                print("Fitting BigMartPreprocessor...")
        
                # Create a copy to avoid modifying original data
                data = X.copy()
                if y is not None:
                    data['Item_Outlet_Sales'] = y
        
                # Calculate item-level statistics
                print("Computing item-level statistics...")
                self.item_stats['Item_mean'] = data.groupby('Item_Identifier')['Item_Outlet_Sales'].mean()
                self.item_stats['Item_std'] = data.groupby('Item_Identifier')['Item_Outlet_Sales'].std()
                self.item_stats['Item_median'] = data.groupby('Item_Identifier')['Item_Outlet_Sales'].median()
                self.item_stats['Item_count'] = data.groupby('Item_Identifier')['Item_Outlet_Sales'].count()
        
                # Calculate outlet-level statistics  
                print("Computing outlet-level statistics...")
                self.outlet_stats['Outlet_mean'] = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean()
                self.outlet_stats['Outlet_std'] = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].std()
                self.outlet_stats['Outlet_median'] = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].median()
                self.outlet_stats['Outlet_count'] = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].count()
                # self.outlet_stats['Outlet_mode'] = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mode()
        
                # Calculate item type statistics
                print("Computing item type statistics...")
                self.item_stats['ItemType_mean'] = data.groupby('Item_Type')['Item_Outlet_Sales'].mean()
                self.item_stats['ItemType_std'] = data.groupby('Item_Type')['Item_Outlet_Sales'].std()
                self.item_stats['ItemType_median'] = data.groupby('Item_Type')['Item_Outlet_Sales'].median()
        
                # Get categorical columns
                self.categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                                          'Outlet_Location_Type', 'Outlet_Type']
        
                # Store all possible categorical values for consistent dummy encoding
                for col in self.categorical_columns:
                    if col in data.columns:
                        # Get unique values, safely handling any NaN for sorting
                        unique_vals = data[col].unique()
                        # Convert to list and filter out NaN values for sorting
                        unique_vals_list = [str(val) for val in unique_vals if pd.notna(val)]
                        self.dummy_columns[col] = sorted(unique_vals_list)
        
                # Store global statistics for fallback values
                self.global_mean = data['Item_Outlet_Sales'].mean() if y is not None else 0
                self.global_std = data['Item_Outlet_Sales'].std() if y is not None else 0
                self.global_median = data['Item_Outlet_Sales'].median() if y is not None else 0
        
                self.is_fitted = True
                print("BigMartPreprocessor fitted successfully!")
                return self
    
            def transform(self, X, pprint=True):
                """Transform the data using fitted statistics"""
                if not self.is_fitted:
                    raise ValueError("Preprocessor must be fitted before transform")

                if pprint:
                    print("Transforming data with BigMartPreprocessor...")
                data = X.copy()
        
                # Handle missing values with smart imputation strategies
                if pprint:
                    print("Handling missing values with smart imputation...")

                # Smart Item_Weight imputation using hierarchical groupby strategy
                if 'Item_Weight' in data.columns:
                    if pprint:
                        print("  - Imputing Item_Weight using multi-level groupby strategy...")
                
                    # Strategy 1: Try Item_Type + Item_Fat_Content first (most specific)
                    weight_type_fat = data.groupby(['Item_Type', 'Item_Fat_Content'])['Item_Weight'].transform('median')
                    data['Item_Weight'] = data['Item_Weight'].fillna(weight_type_fat)
            
                    # Strategy 2: Fallback to Item_Type only for remaining NaNs
                    weight_type_only = data.groupby('Item_Type')['Item_Weight'].transform('median')
                    data['Item_Weight'] = data['Item_Weight'].fillna(weight_type_only)
            
                    # Strategy 3: Global median as final fallback
                    data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].median())
                    if pprint:
                        print(f"    Item_Weight imputed (remaining NaNs: {data['Item_Weight'].isnull().sum()})")
        
                # Smart Outlet_Size imputation using outlet characteristics
                if 'Outlet_Size' in data.columns:
                    if pprint:
                        print("  - Imputing Outlet_Size using outlet type and location patterns...")
            
                    # Create mapping based on most common size for each combination
                    outlet_size_mapping = {
                        ('Grocery Store', 'Tier 1'): 'Small',
                        ('Grocery Store', 'Tier 2'): 'Small', 
                        ('Grocery Store', 'Tier 3'): 'Small',
                        ('Supermarket Type1', 'Tier 1'): 'Medium',
                        ('Supermarket Type1', 'Tier 2'): 'Small',
                        ('Supermarket Type1', 'Tier 3'): 'High',
                        ('Supermarket Type2', 'Tier 1'): 'Medium',
                        ('Supermarket Type2', 'Tier 2'): 'Medium',
                        ('Supermarket Type2', 'Tier 3'): 'Medium',
                        ('Supermarket Type3', 'Tier 1'): 'Medium',
                        ('Supermarket Type3', 'Tier 2'): 'Medium',
                        ('Supermarket Type3', 'Tier 3'): 'Medium'
                    }
            
                    # Apply the mapping for missing values
                    missing_mask = data['Outlet_Size'].isnull()
                    for idx in data[missing_mask].index:
                        outlet_type = data.loc[idx, 'Outlet_Type']
                        location_type = data.loc[idx, 'Outlet_Location_Type']
                        mapping_key = (outlet_type, location_type)
                
                        if mapping_key in outlet_size_mapping:
                            data.loc[idx, 'Outlet_Size'] = outlet_size_mapping[mapping_key]
                        else:
                            # Fallback to most common size for outlet type
                            if outlet_type == 'Grocery Store':
                                data.loc[idx, 'Outlet_Size'] = 'Small'
                            elif outlet_type in ['Supermarket Type2', 'Supermarket Type3']:
                                data.loc[idx, 'Outlet_Size'] = 'Medium'
                            else:
                                data.loc[idx, 'Outlet_Size'] = 'Medium'  # Default fallback
                    if pprint:
                        print(f"    Outlet_Size imputed (remaining NaNs: {data['Outlet_Size'].isnull().sum()})")
            
                # Additional smart missing value handling for other potential missing columns
                if pprint:
                    print("  - Checking for other missing values...")

                # Handle Item_Visibility if it has zeros (which are essentially missing)
                if 'Item_Visibility' in data.columns:
                    zero_visibility = (data['Item_Visibility'] == 0).sum()
                    if zero_visibility > 0:
                        print(f"    - Found {zero_visibility} zero Item_Visibility values, replacing with Item_Type median...")
                        # Replace zeros with median visibility by Item_Type
                        item_visibility_median = data[data['Item_Visibility'] > 0].groupby('Item_Type')['Item_Visibility'].median()
                        for item_type in data['Item_Type'].unique():
                            mask = (data['Item_Type'] == item_type) & (data['Item_Visibility'] == 0)
                            if mask.any() and item_type in item_visibility_median:
                                data.loc[mask, 'Item_Visibility'] = item_visibility_median[item_type]
                
                        # Final fallback to overall median for any remaining zeros
                        overall_visibility_median = data[data['Item_Visibility'] > 0]['Item_Visibility'].median()
                        data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = overall_visibility_median
                
                        print(f"    Item_Visibility zeros handled (remaining zeros: {(data['Item_Visibility'] == 0).sum()})")
                if pprint:
                    print("     Smart missing value imputation completed!")
        
                # Create new features
                if pprint:
                    print("  Creating engineered features...")

                # Basic feature engineering
                if 'Item_Weight' in data.columns and 'Item_MRP' in data.columns:
                    data['Weight_MRP_Ratio'] = data['Item_Weight'] / (data['Item_MRP'] + 1e-8)
            
                if 'Outlet_Establishment_Year' in data.columns:
                    data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
            
                # Add statistical features with proper NaN handling
                if pprint:
                    print("Adding statistical features...")
        
                # Item-level features
                data['Item_mean'] = data['Item_Identifier'].map(self.item_stats['Item_mean']).fillna(self.global_mean)
                data['Item_std'] = data['Item_Identifier'].map(self.item_stats['Item_std']).fillna(self.global_std)
                data['Item_median'] = data['Item_Identifier'].map(self.item_stats['Item_median']).fillna(self.global_median)
                data['Item_count'] = data['Item_Identifier'].map(self.item_stats['Item_count']).fillna(1)
        
                # Outlet-level features
                data['Outlet_mean'] = data['Outlet_Identifier'].map(self.outlet_stats['Outlet_mean']).fillna(self.global_mean)
                data['Outlet_std'] = data['Outlet_Identifier'].map(self.outlet_stats['Outlet_std']).fillna(self.global_std)
                data['Outlet_median'] = data['Outlet_Identifier'].map(self.outlet_stats['Outlet_median']).fillna(self.global_median)
                data['Outlet_count'] = data['Outlet_Identifier'].map(self.outlet_stats['Outlet_count']).fillna(1)
        
                # Item type features
                data['ItemType_mean'] = data['Item_Type'].map(self.item_stats['ItemType_mean']).fillna(self.global_mean)
                data['ItemType_std'] = data['Item_Type'].map(self.item_stats['ItemType_std']).fillna(self.global_std)
                data['ItemType_median'] = data['Item_Type'].map(self.item_stats['ItemType_median']).fillna(self.global_median)
        
                # Handle categorical variables with consistent dummy encoding
                if pprint:
                    print("Encoding categorical variables...")
        
                # Create dummy variables for categorical columns with consistent columns
                for col in self.categorical_columns:
                    if col in data.columns:
                        # Create dummies for all possible values seen during fit
                        # (NaNs should be imputed by now through your smart imputation)
                        dummies = pd.get_dummies(data[col], prefix=col)
                
                        # Ensure all expected columns are present
                        expected_cols = [f"{col}_{val}" for val in self.dummy_columns[col]]
                        for expected_col in expected_cols:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0
                
                        # Select only the expected columns in the correct order
                        dummies = dummies[expected_cols]
                
                        data = pd.concat([data, dummies], axis=1)
                        data.drop(col, axis=1, inplace=True)
        
                # Drop identifier columns
                identifier_cols = ['Item_Identifier', 'Outlet_Identifier']
                for col in identifier_cols:
                    if col in data.columns:
                        data.drop(col, axis=1, inplace=True)
        
                # Final NaN check and cleanup
                if pprint:
                    print("Final data cleanup...")
        
                # Replace any remaining NaN values with 0
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = data[numeric_cols].fillna(0)
        
                # Replace infinite values with finite values
                data.replace([np.inf, -np.inf], 0, inplace=True)

                if pprint:
                    print(f"Transformation complete! Final shape: {data.shape}")

                return data
    
            def fit_transform(self, X, y=None, pprint=True):
                """Fit and transform in one step"""
                return self.fit(X, y).transform(X, pprint=pprint)