# **Time Series Analysis: Monthly Beer Production (1956-1995)**

## **Overview**
This project focuses on analyzing and forecasting monthly beer production data from 1956 to 1995. Here I will explore the dataset, building and comparing multiple models (from simple statistical models to machine learning and deep learning models), and selecting the best-performing model for forecasting.


## **Dataset**
- **Source**: [Monthly Beer Production Dataset](https://www.kaggle.com/datasets/sergiomora823/monthly-beer-production).
- **Columns**:
  - `Month`: The timestamp (monthly frequency).
  - `Monthly beer production`: The production volume (target variable).
- **Rows**: 477 (monthly data from 1956 to 1995).


## **Steps Performed**
1. **Exploratory Data Analysis (EDA)**:
   - Visualized the time series to identify trends, seasonality, and patterns.
   - Decomposed the series into trend, seasonality, and residuals.
   - Checked for stationarity using the Augmented Dickey-Fuller (ADF) test.

2. **Data Preprocessing**:
   - Handled missing values (if any).
   - Created lag features and rolling statistics for machine learning models.
   - Split the data into training (80%) and testing (20%) sets.

3. **Modeling**:
As the dataset very small around 477 values/ rows. Here We will develop:
   - **Simple Models**: SARIMA
   - **Machine Learning Models**: XGBoost, Random Forest.

4. **Evaluation**:
   - Compared models using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
   - Visualized predictions against actual test data.

5. **Model Selection**:
   - Selected the best-performing model based on evaluation metrics.
   - Fine-tuned the best model using hyperparameter optimization.

## **Results**
- **Best Model**: [The best model is: ].
- **Performance Metrics**:
  - MSE: []
  - RMSE: []
  - MAE: []
- **Visualization**:

## **How to Run the Code**
1. Clone the repository:
   ```bash
   git clone 
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook time_series_analysis.ipynb
   ```

## **Dependencies**
- Python 3.x
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `statsmodels`, `scikit-learn`, `xgboost`
  
## **Future Work**
- Incorporate external variables (e.g., economic indicators) to improve forecasting accuracy.
- Experiment with advanced deep learning architectures like Transformers.
- Deploy the best model as a web application for real-time forecasting.

## **Author**
[Subrat Kumar]  
[w.subrat@gmail.com]  
[https://sites.google.com/view/subratdang/home]
