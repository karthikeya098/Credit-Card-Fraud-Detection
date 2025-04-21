import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def main():
    # Load dataset
    print("🔄 Loading dataset...")
    df = pd.read_csv("fraudTest.csv")

    print("\n📌 Column Names:")
    print(df.columns)

    print("\n📊 Class distribution:")
    print(df['is_fraud'].value_counts())

    # Feature Scaling for numerical features
    print("\n🔧 Scaling numeric columns...")
    scaler = StandardScaler()
    df['amt'] = scaler.fit_transform(df[['amt']])
    df['unix_time'] = scaler.fit_transform(df[['unix_time']])

    #  Drop irrelevant columns
    print("\n🧹 Dropping unnecessary columns...")
    drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
                 'job', 'dob', 'trans_num']
    df = df.drop(columns=drop_cols)

    #  Define features and labels
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Train-test split
    print("\n🧪 Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Balance the training data using SMOTE
    print("\n⚖️ Balancing training data with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    print("\n✅ Balanced training set class distribution:")
    print(y_train_balanced.value_counts())

    # Train the model
    print("\n🚀 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    #  Predict on test data
    print("\n🔍 Predicting on test data...")
    y_pred = model.predict(X_test)
    y_actual = y_test.copy()

    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_actual, y_pred))

    print("\n✅ Classification Report:")
    print(classification_report(y_actual, y_pred, digits=4))

    # Visualizing Fraud vs Not Fraud Trends
    print("\n📊 Visualizing Fraud vs Non-Fraud Transactions...")

    # Add back 'hour' column from 'unix_time'
    X_test_plot = X_test.copy()
    X_test_plot['is_fraud'] = y_test
    X_test_plot['hour'] = pd.to_datetime(df['unix_time'], unit='s').dt.hour[:len(X_test_plot)].values

    plt.figure(figsize=(10, 5))
    sns.histplot(data=X_test_plot, x='amt', hue='is_fraud', bins=50, kde=True, palette='coolwarm')
    plt.title("Transaction Amount Distribution (Fraud vs Non-Fraud)")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.legend(title="Is Fraud", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.countplot(data=X_test_plot, x='hour', hue='is_fraud', palette='coolwarm')
    plt.title("Transaction Hour Distribution (Fraud vs Non-Fraud)")
    plt.xlabel("Hour of Transaction")
    plt.ylabel("Count")
    plt.legend(title="Is Fraud", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

    # Show misclassified transactions
    print("\n🔍 Misclassified Transactions:")
    misclassified_indexes = y_actual[y_actual != y_pred].index
    misclassified_samples = X_test.loc[misclassified_indexes]
    actual_labels = y_actual.loc[misclassified_indexes]
    predicted_labels = pd.Series(y_pred, index=y_actual.index).loc[misclassified_indexes]

    if misclassified_samples.empty:
        print("✅ No misclassifications in this test sample.")
    else:
        result = misclassified_samples.copy()
        result['Actual'] = actual_labels
        result['Predicted'] = predicted_labels
        print(result.head(10))  # Display top 10 misclassified rows

    sys.stdout.flush()


# Run the program
if __name__ == "__main__":
    main()
