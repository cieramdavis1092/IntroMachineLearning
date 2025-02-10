# IntroMachineLearning
Homework for GitHub

# Load dataset
breast_cancer = datasets.load_breast_cancer ()
X = data.data
y = data. target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Running Multiple Models to Find the Best One
models = {
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=10000),
    'SVC': SVC(gamma='scale')
}

# K-Fold Cross-Validation
kfold = KFold(n_splits=10, random_state=11, shuffle=True)

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, kfold):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} - Accuracy: {accuracy:.4f}")

  # Display confusion matrix and classification report
  cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}\nClassification Report:\n{classification_report(y_test, y_pred)}\n")

 # Visualize confusion matrix
  plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=breast_cancer.target_names, yticklabels=breast_cancer.target_names)
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Cross-Validation
score = cross_value_score(model, X, y, cv=kfold)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f}")

    # Evaluate all models
for model in model.num():
    evaluate_model(model, X_train, y_train, X_test, y_test, kfold)
