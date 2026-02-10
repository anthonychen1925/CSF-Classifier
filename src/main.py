import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from model import RandomForestModel
from data_utils import load_data, preprocess_data
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the gene expression data
    data = load_data('../Data/filter_combined_pseudobulk.csv')
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    n_runs = 100
    accuracies = []
    for i in range(n_runs):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Initialize the Random Forest model
        rf_model = RandomForestModel()
        
        # Train the model
        rf_model.train_model(X_train, y_train)
        
        # Make predictions
        predictions = rf_model.predict(X_test)

        # Print the true phenotypes and predictions
        print("True Phenotpye:", y_test.values)
        print("Predictions:", predictions)

        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.2f}")

        '''
        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=rf_model.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        '''

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Mean accuracy over {n_runs} runs: {mean_acc:.2f} ± {std_acc:.2f}")
    plt.figure()
    plt.hist(accuracies, bins=5, color='skyblue', edgecolor='black')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Accuracy Distribution over {n_runs} runs')
    plt.xlim(0, 1)
    plt.show()

if __name__ == "__main__":
    main()