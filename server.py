from dataclasses import dataclass, field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
import os
import io

# Initialize FastMCP server
mcp = FastMCP("linear-regression-improved")

@dataclass
class DataContext:
    """
    A class to store and manage the DataFrame in the context.
    
    This class holds the dataset and its original, unprocessed state, allowing for
    re-runs of different preprocessing steps without reloading the file.
    """
    _data: pd.DataFrame = None
    _original_data: pd.DataFrame = None

    def set_data(self, new_data: pd.DataFrame):
        """
        Method to set or update the data, also backing up the original.
        """
        self._original_data = new_data.copy()
        self._data = new_data.copy()

    def get_data(self) -> pd.DataFrame:
        """
        Method to get the current state of the data.
        """
        return self._data

    def reset_data(self):
        """
        Resets the data to its original state after file upload.
        """
        if self._original_data is not None:
            self._data = self._original_data.copy()
            return "Data has been reset to its original state."
        return "No original data available to reset to."

    def is_data_loaded(self) -> bool:
        """
        Checks if data has been loaded into the context.
        """
        return self._data is not None

# Initialize the DataContext instance globally
context = DataContext()

@mcp.tool()
def upload_file(path: str) -> str:
    """
    Reads a CSV file from the given path and stores it in the context.

    Args:
        path (str): The absolute path to the .csv file.

    Returns:
        str: A confirmation message with the shape of the loaded data or an error message.
    """
    if not os.path.exists(path):
        return f"Error: The file at '{path}' does not exist."

    if not path.lower().endswith('.csv'):
        return "Error: The file must be a CSV file."

    try:
        data = pd.read_csv(path)
        context.set_data(data)
        return f"Data successfully loaded. Shape: {data.shape}"
    except pd.errors.EmptyDataError:
        return "Error: The provided CSV file is empty."
    except Exception as e:
        return f"An unexpected error occurred while reading the CSV: {str(e)}"

@mcp.tool()
def get_data_head(rows: int = 5) -> str:
    """
    Returns the first few rows of the dataset as a string.

    Args:
        rows (int): The number of rows to display.

    Returns:
        str: The first 'n' rows of the DataFrame or an error message.
    """
    if not context.is_data_loaded():
        return "Error: No data loaded. Please use 'upload_file' first."
    return context.get_data().head(rows).to_string()

@mcp.tool()
def get_info() -> str:
    """
    Provides a summary of the DataFrame, including data types and null values.

    Returns:
        str: A string containing the DataFrame's info.
    """
    if not context.is_data_loaded():
        return "Error: No data loaded. Please use 'upload_file' first."
    
    buffer = io.StringIO()
    context.get_data().info(buf=buffer)
    return buffer.getvalue()

@mcp.tool()
def preprocess_data() -> str:
    """
    Handles missing values in the dataset by imputing them.
    - Fills numerical columns with the mean.
    - Fills categorical columns with the mode.
    
    This operation modifies the data in context.

    Returns:
        str: A summary of the actions taken.
    """
    if not context.is_data_loaded():
        return "Error: No data loaded. Please use 'upload_file' first."

    data = context.get_data()
    cols_changed = []

    # Impute numerical columns
    for col in data.select_dtypes(include=np.number).columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mean(), inplace=True)
            cols_changed.append(col)

    # Impute categorical columns
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
            cols_changed.append(col)

    if not cols_changed:
        return "No missing values found in the dataset."
        
    return f"Missing values handled in the following columns: {', '.join(cols_changed)}"

@mcp.tool()
def encode_categorical_features(method: str = 'one_hot') -> str:
    """
    Encodes all categorical features using the specified method.
    This operation modifies the data in context.

    Args:
        method (str): The encoding method to use ('one_hot' or 'label').

    Returns:
        str: A confirmation of the encoding operation.
    """
    if not context.is_data_loaded():
        return "Error: No data loaded. Please use 'upload_file' first."

    data = context.get_data()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    if not len(categorical_cols):
        return "No categorical columns found to encode."

    if method == 'label':
        encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = encoder.fit_transform(data[col])
        msg = f"Label encoded the following columns: {', '.join(categorical_cols)}"

    elif method == 'one_hot':
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        context.set_data(data)
        msg = f"One-hot encoded the categorical columns. New shape: {data.shape}"

    else:
        return "Error: Invalid encoding method. Choose 'one_hot' or 'label'."

    return msg

@mcp.tool()
def train_linear_regression_model(output_column: str) -> str:
    """
    Trains a linear regression model using a robust pipeline to prevent data leakage.
    This function performs the following steps:
    1. Identifies categorical and numerical features.
    2. Splits the data into training and test sets.
    3. Creates a preprocessing pipeline that scales numerical features and one-hot encodes categorical features.
    4. Fits the pipeline on the training data and transforms both training and test sets.
    5. Trains a Linear Regression model.
    6. Evaluates the model on the test set and returns the Root Mean Squared Error (RMSE).

    Args:
        output_column (str): The name of the target variable column.

    Returns:
        str: The RMSE of the trained model or an error message.
    """
    if not context.is_data_loaded():
        return "Error: No data loaded. Please use 'upload_file' first."

    try:
        # Use the original, unprocessed data for a clean training pipeline
        data = context._original_data.copy()
        
        if output_column not in data.columns:
            return f"Error: '{output_column}' column not found in the dataset."

        X = data.drop(columns=[output_column])
        y = data[output_column]

        # Identify feature types
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Split data before any preprocessing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create preprocessing pipelines for numerical and categorical features
        # This prevents data leakage by fitting transformers only on the training data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        # Create a preprocessor object using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Create the full model pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Train the model
        model_pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model_pipeline.predict(X_test)

        # Calculate and return RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return f"Model trained successfully using a robust pipeline. Test Set RMSE: {rmse:.4f}"

    except Exception as e:
        return f"An error occurred during model training: {str(e)}"

def main():
    """
    Main function to run the FastMCP server.
    """
    print("Starting Linear Regression MCP Server...")
    mcp.run()

if __name__ == "__main__":
    main()
