
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for time series data"""

    def __init__(self, X, y=None):
        """
        Initialize dataset

        Args:
            X: Input sequences of shape [n_sequences, seq_len, n_features]
            y: Target sequences of shape [n_sequences, horizon, n_features]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation input sequences
        y_val: Validation target sequences
        batch_size: Batch size

    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation (if validation data provided)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create validation dataloader if validation data provided
    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, val_dataloader

    return train_dataloader, None


# Full training loop example with AMP
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, 
                num_epochs=10, device='cuda', use_amp=True, patience=5, kl_weight=1.0):
    """
    Complete training loop with AMP and early stopping
    
    Args:
        model: Seq2Seq model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Maximum number of epochs to train
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        patience: Early stopping patience
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    print(f"Training with {'AMP' if use_amp else 'full precision'}")
    
    for epoch in range(num_epochs):
        # Training phase
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, use_amp, kl_weight)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss = evaluate(model, val_dataloader, criterion, device, use_amp, kl_weight)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            #print(f"Validation loss improved, saving model")
        else:
            epochs_without_improvement += 1
            #print(f"Validation loss did not improve for {epochs_without_improvement} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses