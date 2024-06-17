import pandas as pd
import numpy as np

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
    age = int(input("Enter patient age: "))
    sex = input("Enter pacient sex (M/F): ")
    bp = input("Enter patient blood pressure (HIGH/NORMAL): ")
    cholesterol = input("Enter patient cholesterol (HIGH/NORMAL): ")
    sodium = float(input("Enter patient sodium level: "))
    
    sample = {
        "Age": age,
        "Sex": sex,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Na_to_K": sodium
    }
    
    return sample

if __name__ == "__main__":
    
    file_path = "drug200.csv"
    data = pd.read_csv(file_path)
    
    features = data.columns[:-1]
    target = "Drug"

    decision_tree = ID3(data, data, features, target)

    sample = get_user_input()
    prediction = predict(decision_tree, sample)

    print("The predicted drug for the patient is: ", prediction)