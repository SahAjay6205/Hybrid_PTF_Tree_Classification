import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import TreePCDataset
from src.model import HybridPTF

# --- Training Configuration ---
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 4  # (Quercus rubra, Pinus strobus, Acer saccharum, Betula papyrifera)
NUM_POINTS = 4096 # (K)
DATA_ROOT = './data'

def main():
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Datasets
    # The paper uses 70% Train, 15% Val, 15% Test
    # We use our dummy dataset for all splits for this example
    print("Loading datasets...")
    train_dataset = TreePCDataset(data_root=DATA_ROOT, split='train', num_points=NUM_POINTS)
    val_dataset = TreePCDataset(data_root=DATA_ROOT, split='validation', num_points=NUM_POINTS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Model
    print("Building Hybrid PTF model...")
    model = HybridPTF(num_classes=NUM_CLASSES, num_points=NUM_POINTS).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # 3. Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # The paper uses a ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # 4. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (points, traits, labels) in enumerate(train_loader):
            # Move data to device
            points, traits, labels = points.to(device), traits.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(points, traits)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, traits, labels in val_loader:
                points, traits, labels = points.to(device), traits.to(device), labels.to(device)
                
                outputs = model(points, traits)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Step the scheduler on the validation loss
        scheduler.step(avg_val_loss)

    print("Finished Training.")
    
    # Save the trained model
    torch.save(model.state_dict(), 'hybrid_ptf_model.pth')
    print("Model saved to hybrid_ptf_model.pth")

if __name__ == "__main__":
    main()
