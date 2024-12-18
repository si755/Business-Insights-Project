# Data Pipeline Script (`data_pipeline.py`)
import pandas as pd
import snowflake.connector

# Snowflake connection setup
def connect_snowflake():
    conn = snowflake.connector.connect(
        user='your_user',
        password='your_password',
        account='your_account'
    )
    return conn

# Load data into Snowflake
def load_data_to_snowflake(csv_file, table_name, conn):
    data = pd.read_csv(csv_file)
    cursor = conn.cursor()
    
    for _, row in data.iterrows():
        cursor.execute(
            f"INSERT INTO {table_name} (column1, column2) VALUES (%s, %s)",
            row.values
        )
    conn.close()

# Run pipeline
if __name__ == "__main__":
    conn = connect_snowflake()
    load_data_to_snowflake('data/historical_data_sample.csv', 'sales_data', conn)
    print("Data successfully loaded to Snowflake!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data/historical_data_sample.csv')

# Data preparation
X = data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature names
y = data['target']  # Replace with target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save model
import joblib
joblib.dump(model, 'models/random_forest_model.pkl')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/historical_data_sample.csv')

# Sales trends visualization
sns.lineplot(data=data, x='date', y='sales')
plt.title('Sales Trend Over Time')
plt.savefig('visualizations/sales_trend.png')
plt.show()

# Customer behavior analysis
sns.barplot(data=data, x='customer_segment', y='purchase_amount')
plt.title('Customer Segment vs Purchase Amount')
plt.savefig('visualizations/customer_behavior.png')
plt.show()

