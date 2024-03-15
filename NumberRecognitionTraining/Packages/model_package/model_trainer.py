import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from .models import *

class ModelTrainer(object):
    def __init__(self, train_data, train_labels, test_data_1, test_labels_1, test_data_2, test_labels_2, weight):
        # Convert list of NumPy arrays to PyTorch tensors
        images_tensor = torch.stack([torch.from_numpy(img).float() for img in train_data])
        labels_tensor = torch.from_numpy(np.array(train_labels, dtype=int)).long()
        test_images_tensor = torch.stack([torch.from_numpy(img).float() for img in test_data_1])
        test_labels_tensor = torch.from_numpy(np.array(test_labels_1, dtype=int)).long()
        sudoku_images_tensor = torch.stack([torch.from_numpy(img).float() for img in test_data_2])
        sudoku_labels_tensor = torch.from_numpy(np.array(test_labels_2, dtype=int)).long()

        # Create a custom dataset using TensorDataset
        data_set = TensorDataset(images_tensor, labels_tensor)
        test_data_set = TensorDataset(test_images_tensor, test_labels_tensor)
        test_data_2_set = TensorDataset(sudoku_images_tensor, sudoku_labels_tensor)

        # Create a DataLoader
        batch_size = 64
        self.data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)
        self.test_data_2_loader = DataLoader(test_data_2_set, batch_size=batch_size, shuffle=True)
        self.w = weight
    
    def Test(self,model,device):
        # Testing the model
        correct = 0
        total = 0
        correct_sudoku = 0
        total_sudoku = 0
        with torch.no_grad():
            for data in self.test_data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            for data in self.test_data_2_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_sudoku += labels.size(0)
                correct_sudoku += (predicted == labels).sum().item()
        return (correct / total,correct_sudoku / total_sudoku)
    def OneSeedProgram(self,i,epochs, decision_mode = "both", model_mode = 'ConvolutionalNN', width = 28):
        if not ((decision_mode=="both")|(decision_mode=="sudoku_only")):
            print("Wrong optional parameter: decision_mode.")
            return
        if not ((model_mode=="ConvolutionalNN")|(model_mode=="ComplexNN")):
            print("Wrong optional parameter: model_mode.")
            return
        # Set seed

        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        device = torch.device(dev)

        torch.manual_seed(i)

        # Initialize the neural network, loss function, and optimizer
        if model_mode=="ConvolutionalNN":
            model = ConvolutionalNN(width)
        elif model_mode=="ComplexNN":
            model = ComplexNN(width)
        model = model.to(device)
        weight = torch.FloatTensor(self.w).to(device)
        criterion = nn.CrossEntropyLoss(weight = weight)
        # Define your optimizer with L2 regularization (weight decay)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        #optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_model = None
        best_score = 0
        best_score_test = 0
        best_score_sudoku = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for data in self.data_loader:
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(self.data_loader)}")
            (acc,acc_sud) = self.Test(model,device)
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy on the test set: {acc * 100:.2f}%")
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy on the sudoku set: {acc_sud * 100:.2f}%")
            if decision_mode == "both":
                if min(acc,acc_sud)>best_score:
                    best_score = min(acc,acc_sud)
                    best_score_test = acc
                    best_score_sudoku = acc_sud
                    best_model = model
            elif decision_mode == "sudoku_only":
                if acc_sud>best_score:
                    best_score = acc_sud
                    best_score_test = acc
                    best_score_sudoku = acc_sud
                    best_model = model
        print(f"Best accuracy on the test set: {best_score_test * 100:.2f}%")
        print(f"Best accuracy on the sudoku set: {best_score_sudoku * 100:.2f}%")
        return (best_score_test, best_score_sudoku ,best_model.to(torch.device("cpu")))