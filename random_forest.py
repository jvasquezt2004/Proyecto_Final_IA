import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples == 0 or depth == self.max_depth:
            most_common_label = np.bincount(y).argmax()
            return most_common_label
        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            most_common_label = np.bincount(y).argmax()
            return most_common_label
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return (best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        m, n = X.shape
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_feat, best_thresh = None, None
        for feat in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feat], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = (thresholds[i] + thresholds[i - 1]) / 2
        return best_feat, best_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            if inputs[node[0]] <= node[1]:
                node = node[2]
            else:
                node = node[3]
        return node
    
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return mode(tree_preds, axis=0)[0].flatten()
    
def get_patient_data():
    # Ingresar las características del paciente
    age = int(input("Ingrese la edad del paciente: "))
    sex = input("Ingrese el sexo del paciente (F/M): ")
    bp = input("Ingrese la presión sanguínea del paciente (HIGH/LOW/NORMAL): ")
    cholesterol = input("Ingrese el nivel de colesterol del paciente (HIGH/NORMAL): ")
    na_to_k = float(input("Ingrese la relación Na_to_K del paciente: "))

    # Crear un DataFrame con los datos ingresados
    patient_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'Na_to_K': [na_to_k]
    })

    # Aplicar las mismas transformaciones que en los datos de entrenamiento
    categorical_columns = patient_data.select_dtypes(include=['object']).columns
    patient_data_encoded = pd.get_dummies(patient_data, columns=categorical_columns, drop_first=True)

    # Asegurarse de que las columnas del DataFrame del paciente coincidan con las del conjunto de entrenamiento
    missing_cols = set(X_train.columns) - set(patient_data_encoded.columns)
    for col in missing_cols:
        patient_data_encoded[col] = 0
    patient_data_encoded = patient_data_encoded[X_train.columns]

    return patient_data_encoded.values

if __name__ == "__main__":
    file_path = "drug200.csv"
    data = pd.read_csv(file_path)

    categorical_columns = data.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop('Drug')
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    le = LabelEncoder()
    data_encoded['Drug'] = le.fit_transform(data['Drug'])

    X = data_encoded.drop('Drug', axis=1)
    y = data_encoded['Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X_train.values, y_train.values)

    # Realizar predicciones
    y_pred = rf.predict(X_test.values)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    patient_data = get_patient_data()

    # Predecir el mejor medicamento para el paciente
    predicted_drug = rf.predict(patient_data)
    predicted_drug_label = le.inverse_transform(predicted_drug)[0]

    print(f'El mejor medicamento para el paciente es: {predicted_drug_label}')