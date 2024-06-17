import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Definición de las funciones y clases del modelo

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="Drug"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

def ID3(data, original_data, features, target_attribute_name="Drug", parent_node_class=None):
    if len(data) == 0:
        return parent_node_class
    elif len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

def predict(tree, sample):
    for key in tree.keys():
        value = sample[key]
        if value in tree[key]:
            subtree = tree[key][value]
            if isinstance(subtree, dict):
                return predict(subtree, sample)
            else:
                return subtree
        else:
            closest_value = min(tree[key].keys(), key=lambda x:abs(x-value))
            subtree = tree[key][closest_value]
            if isinstance(subtree, dict):
                return predict(subtree, sample)
            else:
                return subtree

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

# Funciones de la interfaz gráfica

def predict_with_decision_tree(sample):
    prediction = predict(decision_tree, sample)
    messagebox.showinfo("Predicción Árbol de Decisión", f"Resultado: {prediction}")

def predict_with_random_forest(sample):
    patient_data = pd.DataFrame([sample])
    categorical_columns = patient_data.select_dtypes(include=['object']).columns
    patient_data_encoded = pd.get_dummies(patient_data, columns=categorical_columns, drop_first=True)
    missing_cols = set(X_train.columns) - set(patient_data_encoded.columns)
    for col in missing_cols:
        patient_data_encoded[col] = 0
    patient_data_encoded = patient_data_encoded[X_train.columns]
    predicted_drug = rf.predict(patient_data_encoded.values)
    predicted_drug_label = le.inverse_transform(predicted_drug)[0]
    messagebox.showinfo("Predicción Bosque Aleatorio", f"El mejor medicamento para el paciente es: {predicted_drug_label}")

def get_features_and_predict(model):
    try:
        sample = {
            "Age": int(entry_age.get()),
            "Sex": entry_sex.get(),
            "BP": entry_bp.get(),
            "Cholesterol": entry_cholesterol.get(),
            "Na_to_K": float(entry_sodium.get())
        }
        if model == 'decision_tree':
            predict_with_decision_tree(sample)
        elif model == 'random_forest':
            predict_with_random_forest(sample)
    except ValueError:
        messagebox.showerror("Entrada inválida", "Por favor, ingrese valores numéricos válidos.")

# Leer y preparar los datos

file_path = "drug200.csv"
data = pd.read_csv(file_path)

features = data.columns[:-1]
target = "Drug"

decision_tree = ID3(data, data, features, target)

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

# Crear la interfaz gráfica

root = tk.Tk()
root.title("Predicción de Modelos")

tk.Label(root, text="Edad:").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sexo (F/M):").grid(row=1, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=1, column=1)

tk.Label(root, text="Presión (HIGH/LOW/NORMAL):").grid(row=2, column=0)
entry_bp = tk.Entry(root)
entry_bp.grid(row=2, column=1)

tk.Label(root, text="Colesterol (HIGH/NORMAL):").grid(row=3, column=0)
entry_cholesterol = tk.Entry(root)
entry_cholesterol.grid(row=3, column=1)

tk.Label(root, text="Na_to_K:").grid(row=4, column=0)
entry_sodium = tk.Entry(root)
entry_sodium.grid(row=4, column=1)

tk.Button(root, text="Predecir con Árbol de Decisión", command=lambda: get_features_and_predict('decision_tree')).grid(row=5, column=0)
tk.Button(root, text="Predecir con Bosque Aleatorio", command=lambda: get_features_and_predict('random_forest')).grid(row=5, column=1)

root.mainloop()
