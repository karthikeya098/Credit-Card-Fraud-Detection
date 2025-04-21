# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning. A Random Forest Classifier is used, along with SMOTE to handle class imbalance. The model analyzes features from real transaction data to predict fraud and includes insights through data visualizations.

---

## 📂 Dataset

The dataset used in this project contains transaction-level data with features such as transaction amount, time, location, and whether a transaction was fraudulent.

🔗 **Download Dataset Here**: [Fraud Detection Dataset (fraudTest.csv)] (https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)

---

## 🛠️ Technologies Used

- Python 3.x  
- pandas, seaborn, matplotlib  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- Jupyter Notebook / PyCharm (any IDE)

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Feature scaling of numerical values (`amt`, `unix_time`)
   - Dropping irrelevant columns
   - Label extraction and train-test split

2. **Balancing Classes**
   - Fraudulent transactions are rare.
   - SMOTE is used to oversample minority class (fraud cases)

3. **Model Training**
   - A Random Forest Classifier is trained on the balanced data
   - Model is tested on real unbalanced test data

4. **Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)
   - Visualization of Fraud vs Non-Fraud

---

## 📊 Visualizations

- **Transaction Amount Distribution (Fraud vs Non-Fraud)**
- **Transaction Hour Distribution (Fraud vs Non-Fraud)**

These plots help understand how fraudulent activities vary over time and amounts.

---

## 📌 Results

- The model successfully detects fraudulent transactions with a good balance of precision and recall.
- Misclassified samples are printed for analysis.

---

## 🚀 How to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/karthikeya098/Credit-Card-Fraud-Detection.git
   cd fraud-detection-project
   ```

2. **Install Required Packages**  
   Make sure you have Python 3.x installed. Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**  
   Download the `fraudTest.csv` file from the  [Fraud Detection Dataset (fraudTest.csv)] (https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
4. **Run the Main Script**  
   ```bash
   credit_fraud.py
   ```

5. **View Visualizations and Output**  
   - The script will display:
     - Class distribution
     - Evaluation metrics
     - Transaction amount distribution by fraud status
     - Hour-wise fraud distribution
     - Misclassified transactions

