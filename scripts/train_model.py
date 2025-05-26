"""
train_model.py

Defines and trains a custom ResNet18 model for classifying returned clothing items
into main and subcategories. Includes data augmentation, training loop, evaluation,
and saving the final model. Also provides functions for model setup and batch-wise
inference.

Functions:
- CustomResNet18: Modified ResNet18 architecture with dual heads.
- setup_pretrained_model(): Initializes model, dataloaders, loss, optimizer, scheduler.
- train(): Trains the model for multiple epochs with logging.
- evaluate(): Evaluates model performance on the test set.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

num_main_categories = 3 
num_sub_categories = 6 

# Custom ResNet with two output classes (main category and sub-category)
class CustomResNet18(nn.Module):
    def __init__(self, num_main_categories, num_sub_categories):
        super(CustomResNet18, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1') 
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.main_fc = nn.Linear(in_features, num_main_categories)
        self.sub_fc = nn.Linear(in_features, num_sub_categories)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        main_output = self.main_fc(x)
        sub_output = self.sub_fc(x)
        return main_output, sub_output

# Function to set up pre-trained model
def setup_pretrained_model():
    """
    Initializes a custom ResNet18 model with dual output heads.

    Applies data augmentation and normalization, sets up dataloaders,
    loss functions, optimizer, scheduler, and moves the model to the appropriate device.

    Returns:
        model (nn.Module): The initialized model.
        criterion_main (Loss): Loss for main category classification.
        criterion_sub (Loss): Loss for subcategory classification.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LRScheduler): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
        device (torch.device): Device used for computation.
    """
    
    # Define the image transformations: resizing, converting to tensor, and normalizing
    transform = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    
    # Load datasets with transformations (train and test sets)
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = ImageFolder(root=TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the custom model with main categories and sub-categories
    model = CustomResNet18(num_main_categories=num_main_categories, num_sub_categories=num_sub_categories)
    
    # Unfreeze the backbone
    for param in model.resnet.parameters():
        param.requires_grad = True
        
    # Optionally freeze batch norm layers (prevent overfitting)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)  # Keep batch norm layers in inference mode during training
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define class weights based on data distribution
    main_class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    sub_class_weights = torch.tensor([1.0] * num_sub_categories).to(device)

    # Define weighted loss function for better balance
    criterion_main = nn.CrossEntropyLoss(weight=main_class_weights, label_smoothing=0.1)
    criterion_sub = nn.CrossEntropyLoss(weight=sub_class_weights, label_smoothing=0.1)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Add a learning rate scheduler to improve convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    return model, criterion_main, criterion_sub, optimizer, scheduler, train_loader, test_loader, device
    
# Now proceed with training the model or using it for inference

def train(model, criterion_main, criterion_sub, optimizer, scheduler, train_loader, device, num_epochs=15):
    """
    Trains the model for main and subcategory classification.

    Runs multiple epochs over the training data, computes loss and accuracy
    for both label levels, updates weights, and applies learning rate scheduling.

    Args:
        model (nn.Module): The model to be trained.
        criterion_main (Loss): Loss function for main categories.
        criterion_sub (Loss): Loss function for subcategories.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LRScheduler): Learning rate scheduler.
        train_loader (DataLoader): Training data loader.
        device (torch.device): Device to run the training on.
        num_epochs (int, optional): Number of training epochs. Defaults to 15.
    """
    
    print("Starting training...")
    
    # Define label mappings
    combined_mapping = {
        0: ("bottomwear", "pants"),
        1: ("bottomwear", "shorts"),
        2: ("footwear", "heels"),
        3: ("footwear", "sneakers"),
        4: ("upperwear", "jacket"),
        5: ("upperwear", "shirt"),
        }

    for label, (main_category, sub_category) in combined_mapping.items():
        print(f"  Label {label}: {main_category} → {sub_category}")
        
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_main = 0
        correct_sub = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            
            # Map 6 subcategory labels (0–5) to 3 main categories (0–2)
            main_labels = (labels // 2).to(device)
            sub_labels = labels.to(device)
            
            optimizer.zero_grad()
            main_output, sub_output = model(images)
            
            # Compute the loss separately for main and subcategories
            main_loss = criterion_main(main_output, main_labels)
            sub_loss = criterion_sub(sub_output, sub_labels)
            loss = main_loss + sub_loss
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
            total += labels.size(0)
            correct_main += (main_output.argmax(1) == main_labels).sum().item()
            correct_sub += (sub_output.argmax(1) == sub_labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        main_accuracy = 100 * correct_main / total
        sub_accuracy = 100 * correct_sub / total
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Main Accuracy: {main_accuracy:.2f}%, Sub Accuracy: {sub_accuracy:.2f}%")
        
        # Step the scheduler
        scheduler.step()
        
    # Save the trained model
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")
    print("Finished Training")

# Evaluating model on test dataset
def evaluate(model, test_loader, device):
    """
    Evaluates the trained model on the test dataset.

    Computes and prints the accuracy for both main and subcategory predictions.

    Args:
        model (nn.Module): Trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
    """
    print("Starting evaluation...")
    
    model.eval()  # Set the model to evaluation mode
    correct_main = 0
    correct_sub = 0
    total = 0

    combined_mapping = {
        0: ("bottomwear", "pants"),
        1: ("bottomwear", "shorts"),
        2: ("footwear", "heels"),
        3: ("footwear", "sneakers"),
        4: ("upperwear", "jacket"),
        5: ("upperwear", "shirt"),
        }

    for label, (main, sub) in combined_mapping.items():
        print(f"  Label {label}: {main} → {sub}")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            main_labels = (labels // 2).to(device)
            sub_labels = labels.to(device)

            main_output, sub_output = model(images)

            # Compare predictions with ground truth
            correct_main += (main_output.argmax(1) == main_labels).sum().item()
            correct_sub += (sub_output.argmax(1) == sub_labels).sum().item()
            total += labels.size(0)

    main_accuracy = 100 * correct_main / total
    sub_accuracy = 100 * correct_sub / total

    print(f"Test Accuracy of the model on the test images: "
          f"Main: {main_accuracy:.2f}%, Subclass: {sub_accuracy:.2f}%")

if __name__ == '__main__':
    #Call the function to get the model, criterion, optimizer, and data loaders
    model, criterion_main, criterion_sub, optimizer, scheduler, train_loader, test_loader, device= setup_pretrained_model()
    
    # Train the model
    train(model, criterion_main, criterion_sub, optimizer, scheduler, train_loader, device, num_epochs=15) # Call the train function
    
    # After training, evaluate the model on the test dataset
    evaluate(model, test_loader, device,)