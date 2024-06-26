import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

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
    # Caso base: Si el conjunto de datos está vacío, devolver el nodo padre
    if len(data) == 0:
        return parent_node_class
    
    # Caso base: Si todos los elementos tienen la misma clase, devolver esa clase
    elif len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Caso base: Si no hay más características, devolver el modo del nodo padre
    elif len(features) == 0:
        return parent_node_class
    
    # Caso recursivo
    else:
        # Establecer el nodo padre
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Seleccionar la característica que tiene la mayor ganancia de información
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Crear la estructura del árbol
        tree = {best_feature: {}}
        
        # Remover la mejor característica de la lista de características
        features = [i for i in features if i != best_feature]
        
        # Crecer el árbol para cada valor de la característica seleccionada
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
            
def get_user_input():
    age = int(age_entry.get())
    sex = sex_entry.get()
    bp = bp_entry.get()
    cholesterol = cholesterol_entry.get()
    sodium = float(sodium_entry.get())
    
    sample = {
        "Age": age,
        "Sex": sex,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Na_to_K": sodium
    }
    
    return sample

def on_predict_button_click():
    sample = get_user_input()
    prediction = predict(decision_tree, sample)
    messagebox.showinfo("Prediction", f"The predicted drug for the patient is: {prediction}")

def accuracy(tree, data):
    correct_predictions = 0
    total_samples = len(data)

    for _, sample in data.iterrows():
        prediction = predict(tree, sample)
        if prediction == sample["Drug"]:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

if __name__ == "__main__":
    file_path = "drug200.csv"
    data = pd.read_csv(file_path)

    features = data.columns[:-1]
    target = "Drug"

    decision_tree = ID3(data, data, features, target)

    # Create the tkinter window
    window = tk.Tk()
    window.title("Drug Recommendation System")

    # Create the input labels and entry fields
    age_label = tk.Label(window, text="Age:")
    age_label.pack()
    age_entry = tk.Entry(window)
    age_entry.pack()

    sex_label = tk.Label(window, text="Sex (M/F):")
    sex_label.pack()
    sex_entry = tk.Entry(window)
    sex_entry.pack()

    bp_label = tk.Label(window, text="Blood Pressure (LOW/HIGH/NORMAL):")
    bp_label.pack()
    bp_entry = tk.Entry(window)
    bp_entry.pack()

    cholesterol_label = tk.Label(window, text="Cholesterol (HIGH/NORMAL):")
    cholesterol_label.pack()
    cholesterol_entry = tk.Entry(window)
    cholesterol_entry.pack()

    sodium_label = tk.Label(window, text="Sodium Level:")
    sodium_label.pack()
    sodium_entry = tk.Entry(window)
    sodium_entry.pack()

    # Create the predict button
    predict_button = tk.Button(window, text="Predict", command=on_predict_button_click)
    predict_button.pack()

    # Run the tkinter event loop
    window.mainloop()

    # Datos de prueba
    test_data = pd.DataFrame({
        "Age": [30, 60, 22, 45, 50],
        "Sex": ["M", "F", "M", "F", "M"],
        "BP": ["HIGH", "NORMAL", "LOW", "NORMAL", "HIGH"],
        "Cholesterol": ["HIGH", "NORMAL", "HIGH", "NORMAL", "HIGH"],
        "Na_to_K": [15, 12, 8, 10, 20],
        "Drug": ["drugY", "drugX", "drugC", "drugX", "drugY"]
    })

    test_accuracy = accuracy(decision_tree, test_data)
    print("The accuracy of the model on test data is: ", test_accuracy)