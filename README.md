# Breast-cancer-diagnosis-
predicting if cancer is malignant or benign
1. Introduction

The goal of this project is to develop a machine learning model to predict whether a breast tumor is malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses. Early and accurate diagnosis of breast cancer is crucial for effective treatment and improving patient outcomes.

2. Data Exploration and Preprocessing

The dataset used for this analysis contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image. The dataset was loaded and explored using Pandas and visualized using Matplotlib and Seaborn.

**Key findings from data exploration:**

* The dataset contains 569 instances with 33 features (including the target variable 'diagnosis').
* The 'diagnosis' column indicates whether the tumor is malignant (M) or benign (B).
* Some features are highly correlated, suggesting potential redundancy.
* The dataset is imbalanced, with more benign cases than malignant cases.

**Preprocessing steps:**

* The 'Unnamed: 32' column was dropped as it contained missing values.
* The 'id' column was dropped as it is not relevant for prediction.
* The 'diagnosis' column was mapped to numerical values (M: 1, B: 0) for model compatibility.
* Highly correlated features ('area_mean', 'perimeter_mean', 'area_worst') were dropped to reduce redundancy.

**3. Model Selection and Training**

Several classification algorithms were considered for this task, including:

* **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies data points based on their proximity to neighboring data points.
* **Logistic Regression:** A linear model used for binary classification problems.

These models were selected due to their simplicity, interpretability, and suitability for binary classification tasks.

**Model training and evaluation:**

* The dataset was split into training and testing sets (80/20 split).
* KNN and Logistic Regression models were trained on the training set.
* Model performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and Jaccard similarity score.

**4. Results and Analysis**

**KNN:**

* Achieved a Jaccard similarity score of approximately [0.91].
* The confusion matrix revealed [ the model shows strong performance with high accuracy, precision, and specificity. The main area for potential improvement is recall for class 1, as there are a few false negatives.].
* The classification report showed [ that the model performed well in distinguishing between the two classes. It excels in predicting both classes, with particularly strong performance in class 1 precision and class 0 recall. The overall metrics suggest the model is reliable and effective for the given classification task.].

**Logistic Regression:**

* Achieved a higher accuracy and F1-score compared to KNN.
* Hyperparameter tuning using GridSearchCV further improved model performance.
* The best hyperparameters were found to be [{'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}].
* The final model achieved [Precision for class 0: 0.96

Recall for class 0: 0.99

F1-score for class 0: 0.97

Precision for class 1: 0.98

Recall for class 1: 0.93

F1-score for class 1: 0.95

Overall Accuracy: 0.96

Macro Average Precision: 0.97

Macro Average Recall: 0.96

Macro Average F1-score: 0.96

Weighted Average Precision: 0.97

Weighted Average Recall: 0.96

Weighted Average F1-score: 0.96].

**5. Conclusion and Recommendations**

Based on the results, the Logistic Regression model with tuned hyperparameters outperformed the KNN model in predicting breast cancer diagnosis. This model can be used as a tool to assist healthcare professionals in making more informed decisions regarding diagnosis and treatment.

**Recommendations for future work:**

* Explore more advanced machine learning algorithms, such as Support Vector Machines (SVM) or Random Forests, to potentially improve predictive performance.
* Address the class imbalance in the dataset using techniques like oversampling or undersampling.
* Incorporate additional features or data sources to enhance the model's predictive power.
* Deploy the model in a clinical setting to evaluate its real-world performance and impact.
