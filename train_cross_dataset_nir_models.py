from operator import le
import os
import sys
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
from modules.dataset import CustomDataset
from modules.networks import CustomVGG19, CustomDenseNet121, CustomResNet152, CustomInception

# Define the paths and other parameters
image_root_dir = './pm-iris-dataset/warsaw-nij-cropped-nir-images/'
train_data_path = './train-test-data-list/dataset-disjoint/warsaw-cropped-NIR-image-metadata.txt'
test_data_path = './train-test-data-list/dataset-disjoint/nij-cropped-NIR-image-metadata.txt'
output_dir = './dataset-disjoint-result/test-on-nij/'

# Model Parameters
model_name = 'vgg'
pretrained = False
solver_name = 'Adam'
batch_size = 20
lr = 0.0001
weight_decay = True

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {device} | Pretrained: {pretrained} | Solver Name: {solver_name} | Weight Decay: {weight_decay}')

# Load pre-trained model
if model_name == "vgg":
    model = CustomVGG19(input_channels=1, pretrained=pretrained, num_classes=1)
    model = model.to(device)
elif model_name == "resnet":    
    model = CustomResNet152(input_channels=1, pretrained=pretrained, num_classes=1)
    model = model.to(device)
elif model_name == "inception":
    model = CustomInception(input_channels=1, pretrained=pretrained, num_classes=1)
    model = model.to(device)
elif model_name == "densenet":
    model = CustomDenseNet121(input_channels=1, pretrained=pretrained, num_classes=1)
    model = model.to(device)
else:
    print("Model not found. Exiting.")
    sys.exit()

print(f'Training Network:\n{model}')

# Define loss function and optimizer
criterion = nn.MSELoss()
if solver_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6 if weight_decay else 0)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6 if weight_decay else 0)

# Define transformations to apply to the images
if model_name == "inception": 
    trainTransform = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
        transforms.ToTensor()
    ])
else:
    trainTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-30, 30)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
    transforms.ToTensor()
    ])

testTransform = transforms.Compose([
    transforms.ToTensor()
])


# Define your train and test datasets
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Create instances of the CustomDataset
train_dataset = CustomDataset(data=train_data, root_dir=image_root_dir, transform=trainTransform)
test_dataset = CustomDataset(data=test_data, root_dir=image_root_dir, transform=testTransform)

# Create DataLoaders for batching and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 500
best_mae = float('inf')
best_model_path = f"{output_dir}{model_name}_NIR_opt_{solver_name}_pret_{str(pretrained)}_wd_{str(weight_decay)}_best_model.pth"

for epoch in range(num_epochs):
    # Training
    model.train()
    total_running_loss = 0.0

    for filenames, inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_running_loss += loss.item()

    # To save validation data output
    val_all_file_names = []
    val_all_actual_values = []
    val_all_predicted_values = []

    # Validation
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        total_mse = 0.0
        total_rmse = 0.0
        total_mae = 0.0

        for filenames, inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            actual = targets.view(-1).detach().cpu().numpy()
            predicted = outputs.view(-1).detach().cpu().numpy()

            total_mse += mean_squared_error(actual, predicted)
            total_rmse += np.sqrt(mean_squared_error(actual, predicted))
            total_mae += mean_absolute_error(actual, predicted)

            # Append current batch values to the lists
            val_all_file_names.extend(filenames)
            val_all_actual_values.extend(actual)
            val_all_predicted_values.extend(predicted)

    train_loss = total_running_loss / len(train_dataloader)
    val_loss = total_val_loss / len(test_dataloader)
    val_mse = total_mse / len(test_dataloader)
    val_rmse = total_rmse / len(test_dataloader)
    val_mae = total_mae / len(test_dataloader)

    # Save best model based on MAE
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.2f} Val Loss: {val_loss:.2f} "
        f"MSE: {val_mse:.2f} RMSE: {val_rmse:.2f} MAE: {val_mae:.2f}")

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Test your model on new data if needed
all_file_names = []
all_actual_values = []
all_predicted_values = []

model.eval()
with torch.no_grad():
    total_val_loss = 0.0
    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0

    for filenames, inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_val_loss += loss.item()

        actual = targets.view(-1).detach().cpu().numpy()
        predicted = outputs.view(-1).detach().cpu().numpy()

        total_mse += mean_squared_error(actual, predicted)
        total_rmse += np.sqrt(mean_squared_error(actual, predicted))
        total_mae += mean_absolute_error(actual, predicted)

        # Append current batch values to the lists
        all_file_names.extend(filenames)
        all_actual_values.extend(actual)
        all_predicted_values.extend(predicted)

val_loss = total_val_loss / len(test_dataloader)
val_mse = total_mse / len(test_dataloader)
val_rmse = total_rmse / len(test_dataloader)
val_mae = total_mae / len(test_dataloader)

print('-------------------------------------')
print('--------Best Model Performance-------')
print('-------------------------------------')
print(f'       MSE: {val_mse:.2f}')
print(f'       RMSE: {val_rmse:.2f}')
print(f'       MAE: {val_mae:.2f}')

# Customize matplotlib and seaborn
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.size'] = 15
plt.rcParams["figure.figsize"] = (8, 6)

# Create a scatter plot with diagonal line
plt.scatter(all_actual_values, all_predicted_values, color='blue', label='Predicted')
plt.plot(all_actual_values, all_actual_values, color='red', label='Actual', linestyle='dashed')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}{model_name}_NIR_plot.pdf", format='pdf', dpi=3000)
plt.clf()

# Create a DataFrame for actual and predicted values
results = pd.DataFrame({'filename': all_file_names, 'Actual': all_actual_values, 'Predicted': all_predicted_values})
# Save the results DataFrame to a text file
results.to_csv(f"{output_dir}{model_name}_NIR_test_result.txt", sep=',', index=False)

