# NFL Yardage Gain Prediction with PySpark

This project uses PySpark to predict the yardage gained on NFL plays using the NFL Big Data Bowl 2020 dataset. The project demonstrates a basic Spark ML pipeline for data preprocessing, feature engineering, model training, and evaluation.
<p align="center">
  <img src="https://github.com/GongiAhmed/NFL-Prediction/blob/main/NFL%20Prediction/images/NFL.png" />
</p>


## Dataset

The project utilizes the NFL Big Data Bowl 2020 dataset from Kaggle ([link to dataset](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/data)). This dataset contains player tracking data, play information, and game conditions for thousands of NFL plays.

## Approach

The notebook `nfl-prediction-with-pyspark.ipynb` follows these steps:

1. **Environment Setup:** Initializes a SparkSession and imports necessary libraries.
2. **Exploratory Data Analysis (EDA):**  Performs basic EDA using PySpark to gain insights into the data. This includes:
    * Checking for missing values.
    * Visualizing the distribution of the target variable ('Yards').
    * Computing correlations between 'Yards' and other numerical features.
    * Exploring categorical features like 'Team' and 'OffenseFormation'.
3. **Data Preparation:**  Preprocesses the data for model training:
    * Handles missing values in numerical features by imputing with the median.
    * Handles missing categorical values by filling with "Unknown".
    * Converts categorical features into numerical representations using StringIndexer.
    * Assembles features into a single vector using VectorAssembler.
    * Scales features using StandardScaler (optional, but often beneficial for linear models).
4. **Model Training:** Trains three regression models:
    * Linear Regression
    * Random Forest Regressor
    * Gradient-Boosted Trees Regressor
5. **Model Evaluation:** Evaluates model performance on a held-out test set using RMSE and R-squared.
6. **Hyperparameter Tuning:** Demonstrates hyperparameter tuning using `CrossValidator` with `ParamGridBuilder` for the Random Forest model to improve performance.
7. **Model Saving and Loading:** Shows how to save the best performing model (tuned Random Forest in this case) and load it for later use.


## Results


The initial results before hyperparameter tuning are as follows:


| Model                    | RMSE    | R²       |
|--------------------------|---------|----------|
| Linear Regression        | 6.40    | 0.011    |
| Random Forest            | 6.39    | 0.016   |
| Gradient Boosted Trees | 6.36    | 0.024  |

After hyperparameter tuning (for Random Forest), the RMSE improved to 6.20 and R² to 0.072.

The main visualization generated is the histogram of the target variable and correlation scatterplots for speed vs. acceleration.


## Further Improvements

* **More Extensive Feature Engineering:** Explore additional features, such as time-based features from 'GameClock', player-specific statistics, or interactions between features.
* **More Advanced Hyperparameter Tuning:** Use a more extensive parameter grid and potentially RandomizedSearchCV for more efficient tuning.
* **Ensemble Methods:** Try other ensemble methods or explore stacking/blending to combine the predictions of multiple models.
* **Feature Selection:** Use feature importance scores from the Random Forest or Gradient Boosted Trees models to select the most relevant features and potentially improve performance and model interpretability.
* **Consider other algorithms:** While random forest was the most successful of the simple algorithms attempted, consider using XGBoost, LightGBM, or CatBoost, which are often top performers with structured datasets.



##  Instructions to Run

1. **Install PySpark:** Install it either locally or use a cloud-based Databricks community edition which has a free tier available.
2. **Download the dataset:** Obtain the NFL Big Data Bowl 2020 dataset from Kaggle and put the `train.csv` file in the correct location in the file system that you're using with your version of Spark.
3. **Run the notebook:** Open and execute the Jupyter notebook `nfl-prediction-with-pyspark.ipynb`. Ensure your Spark environment is configured correctly. Adjust the file path if needed when reading the dataset in Spark.
