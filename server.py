from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
import os

# Initialize FastMCP server
mcp = FastMCP("linear-regression")

@dataclass
class DataContext():
    """
    A class that stores the DataFrame in the context.
    """
    _data: pd.DataFrame = None

    def set_data(self, new_data: pd.DataFrame):
        """
        Method to set or update the data.
        """
        self._data = new_data

    def get_data(self) -> pd.DataFrame:
        """
        Method to get the data from the context.
        """
        return self._data

# Initialize the DataContext instance globally
context = DataContext()

@mcp.tool()
def upload_file(path: str) -> str:
    """
    This function read the csv data and stores it in the class variable.

    Args:
        Absolute path to the .csv file.

    Returns:
        String which shows the shape of the data.
    """

    if not os.path.exists(path):
        return f"Error: The file at '{path}' does not exist."

    # Check if file has a .csv extension
    if not path.lower().endswith('.csv'):
        return "Error: The file must be a CSV file."

    try:
        # Try to read the CSV file using pandas
        data = pd.read_csv(path)
        
        # Store the data in the DataContext class
        context.set_data(data)

        # Store the shape of the data (rows, columns)
        data_shape = context.get_data().shape

        return f"Data successfully loaded. Shape: {data_shape}"
    except Exception as e:
        return f"An unexpected error occured: {str(e)}"
    
@mcp.tool()
def get_columns_info() -> str:
    """
    This function gives information about columns.

    Returns:
        String which contains column names.
    """

    columns = context.get_data().columns

    return ", ".join(columns)

@mcp.tool()
def check_category_columns() -> str:
    """
    This function check if data has categorical columns.

    Returns:
        String which contains list of categorical columns.
    """

    categorical_data = context.get_data().select_dtypes(include=["object", "category"])

    if not categorical_data.empty:
        return f"Data has following categorical columns: {", ".join(categorical_data.columns.to_list())}"
    else:
        return f"Data has no categorical columns."


@mcp.tool()
def label_encode_categorical_columns() -> str:
    """
    This function label encodes all the categorical columns.

    Returns:
        String which confirms success of encoding process.
    """

    categorical_columns = context.get_data().select_dtypes(include=["object", "category"]).columns

    if len(categorical_columns) == 0:
        return "No categorical columns found to encode."

    # Initialize the LabelEncoder
    encoder = LabelEncoder()

    # Iterate over each categorical column and apply label encoding
    for column in categorical_columns:
        context.get_data()[column] = encoder.fit_transform(context.get_data()[column])

    return "All categorical columns have been label encoded successfully."

@mcp.tool()
def train_linear_regression_model(output_column: str) -> str:
    """
    This function trains linear regression model.

    Args:
        Takes input for output column name.

    Returns:
        String which contains the RMSE value.
    """

    try:
        data = context.get_data()

        # Check if the output column exists in the dataset
        if output_column not in data.columns:
            return f"Error: '{output_column}' column not found in the dataset."

        # Prepare the features (X) and target variable (y)
        X = data.drop(columns=[output_column])  # Drop the target column for features
        y = data[output_column]  # The target variable is the output column

        # Split the data into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Initialize the Linear Regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Return the RMSE value
        return f"Model trained successfully. RMSE: {rmse:.4f}"

    except Exception as e:
        return f"An error occurred while training the model: {str(e)}"

if __name__ == "__main__":
    mcp.run()