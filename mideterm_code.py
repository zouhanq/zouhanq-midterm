# Libraries
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt


# Set random seed
np.random.seed(42)


# Load data
data = pd.read_csv("./data/train.csv", usecols=['Id', 'Summary', 'Text', 'Score'])
test_ids = pd.read_csv("./data/test.csv", usecols=['Id'])


# Extract and remove test data
test = data[data['Id'].isin(test_ids['Id'])].copy()
test['Combined_Text'] = test['Summary'].fillna('') + ' ' + test['Text'].fillna('')
train = data[~data['Id'].isin(test_ids['Id'])].copy()


# Drop rows with missing scores and cast 'Score' to int8
train = train.dropna(subset=['Score'])
train['Score'] = train['Score'].astype(np.int8)


# Combine 'Summary' and 'Text'
train['Combined_Text'] = train['Summary'].fillna('') + ' ' + train['Text'].fillna('')


# Sample 30,000 rows for efficiency
train = train.sample(30000, random_state=42)


# Stemming/Lemmatization
stemmer = SnowballStemmer("english")
train['Combined_Text'] = train['Combined_Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
test['Combined_Text'] = test['Combined_Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))


# Feature Engineering
def feature_engineering(df):
    df['Review_Length'] = df['Combined_Text'].str.len()
    df['Word_Count'] = df['Combined_Text'].str.split().str.len()
    df['Exclamation_Count'] = df['Combined_Text'].str.count('!')
    df['Uppercase_Word_Count'] = df['Combined_Text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df['Exclamation_Word_Ratio'] = df['Exclamation_Count'] / (df['Word_Count'] + 1e-6)
    return df


train = feature_engineering(train)
test = feature_engineering(test)


# Define numerical features
features = ['Review_Length', 'Word_Count', 'Exclamation_Count',
            'Uppercase_Word_Count', 'Exclamation_Word_Ratio']


# Prepare numerical feature arrays
X_train_num = train[features].values
X_test_num = test[features].values


# Polynomial Features with degree=3
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_num)
X_test_poly = poly.transform(X_test_num)


# Standardize numerical features
means, stds = np.mean(X_train_poly, axis=0), np.std(X_train_poly, axis=0)
stds[stds == 0] = 1  # Avoid division by zero
X_train_poly = (X_train_poly - means) / stds
X_test_poly = (X_test_poly - means) / stds


# TF-IDF Vectorization with increased features
vectorizer = TfidfVectorizer(
    max_features=5000,  # Increase feature size
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)


tfidf_train = vectorizer.fit_transform(train['Combined_Text'])
tfidf_test = vectorizer.transform(test['Combined_Text'])


# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=120, random_state=42)  # Increased components
X_train_svd = svd.fit_transform(tfidf_train)
X_test_svd = svd.transform(tfidf_test)


# Combine numerical and text features
X_train_combined = np.hstack([X_train_poly, X_train_svd])
X_test_combined = np.hstack([X_test_poly, X_test_svd])


# Define target variable
y_train = train['Score']


# Train/Validation Split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.25, stratify=y_train, random_state=42
)


# Initialize LinearSVC with custom class weights
class_weights = {1: 4, 2: 3, 3: 2, 4: 1.5, 5: 1}
model = LinearSVC(class_weight=class_weights, random_state=42, max_iter=3000)


# GridSearchCV for Hyperparameter Tuning
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)


print("Running GridSearchCV...")
grid_search.fit(X_train_split, y_train_split)


# Best model from GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")


# Evaluate on the validation set
y_val_pred = best_model.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")


# Display confusion matrix
cm = confusion_matrix(y_val_split, y_val_pred, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Validation Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Cross-validation score for stability
cv_scores = cross_val_score(best_model, X_train_combined, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f}")


# Predict on the test set
y_test_pred = best_model.predict(X_test_combined)


# Check the distribution of predictions
print(f"Test Predictions Distribution:\n{pd.Series(y_test_pred).value_counts()}")


# Create submission file
test['Score'] = y_test_pred
submission = test[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)


print("Submission file created successfully.")
