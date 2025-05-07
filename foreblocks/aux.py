import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.amp import autocast, GradScaler
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# Example of evaluation function with AMP support
def evaluate(model, dataloader, criterion, device='cuda', use_amp=True, kl_weight=1.0):
    """
    Evaluate the model with optional Automatic Mixed Precision
    
    Args:
        model: Seq2Seq model
        dataloader: DataLoader providing (src, tgt) pairs
        criterion: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        eval_loss: Average loss for the evaluation
    """
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            if use_amp:
                # Forward pass with mixed precision
                with autocast('cuda'):
                    output = model(src)
                    loss = criterion(output, tgt)
                    if hasattr(model, "_kl") and model._kl is not None:
                        loss += kl_weight * model._kl

            else:
                # Standard forward pass
                output = model(src)
                loss = criterion(output, tgt)
                if hasattr(model, "_kl") and model._kl is not None:
                    loss += kl_weight * model._kl

            # Accumulate loss
            eval_loss += loss.item()
    
    return eval_loss / len(dataloader)


# Example of training function with AMP support
def train_epoch(model, dataloader, optimizer, criterion, device='cuda', use_amp=True, kl_weight=1.0):
    """
    Train the model for one epoch with optional Automatic Mixed Precision
    
    Args:
        model: Seq2Seq model
        dataloader: DataLoader providing (src, tgt) pairs
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if use_amp else None
    
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        if use_amp:
            # Forward pass with mixed precision
            # Forward pass with mixed precision
            with autocast('cuda'):
                output = model(src, tgt)
                loss = criterion(output, tgt)

                # Add KL divergence if applicable
                if hasattr(model, "_kl") and model._kl is not None:
                    loss += model._kl  # or weight it: `+ beta * model._kl`

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights with gradient scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            output = model(src, tgt)
            loss = criterion(output, tgt)

            if hasattr(model, "_kl") and model._kl is not None:
                loss += model._kl

            # Standard backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

