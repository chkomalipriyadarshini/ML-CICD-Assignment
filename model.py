from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X_test, y_test, predictions

if __name__ == "__main__":
    model, accuracy, X_test, y_test, predictions = train_model()
    print("ML Model: Random Forest Classifier")
    print("Dataset: Iris")
    print(f"Model Accuracy: {accuracy:.2%}")
    print("Sample Predictions:", predictions[:5])
