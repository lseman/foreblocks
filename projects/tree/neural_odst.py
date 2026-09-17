# neural_odst.py - Python side neural oblivious decision tree

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ODST(nn.Module):
    """Neural Oblivious Decision Tree from NODE paper."""
    
    def __init__(
        self,
        in_features: int,
        num_trees: int = 64,
        depth: int = 6,
        tree_dim: int = 1,
        flatten_output: bool = True,
        choice_function: str = 'entmax15',
        bin_function: str = 'entmax15',
    ):
        super().__init__()
        self.in_features = in_features
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.flatten_output = flatten_output
        
        # Feature selection: (depth, in_features) per tree
        self.feature_selection_logits = nn.Parameter(
            torch.zeros(num_trees, depth, in_features)
        )
        
        # Thresholds: (depth,) per tree
        self.feature_thresholds = nn.Parameter(torch.zeros(num_trees, depth))
        
        # Temperature for bin function
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, depth))
        
        # Response tensor: (num_trees, 2^depth, tree_dim)
        self.response = nn.Parameter(
            torch.randn(num_trees, 2 ** depth, tree_dim) * 0.01
        )
        
        self.bin_function = self._get_bin_fn(bin_function)
        self.choice_function = self._get_choice_fn(choice_function)
        
        self._initialize_params()
    
    def _get_bin_fn(self, name: str):
        if name == 'entmax15':
            return entmax15_2d
        elif name == 'sparsemax':
            return sparsemax_2d
        else:  # softmax
            return lambda x: torch.softmax(torch.stack([x, torch.zeros_like(x)], -1), -1)[..., 0]
    
    def _get_choice_fn(self, name: str):
        if name == 'entmax15':
            return entmax15
        elif name == 'sparsemax':
            return sparsemax
        else:
            return torch.softmax
    
    def _initialize_params(self):
        """Data-aware initialization happens in initialize() method."""
        nn.init.uniform_(self.feature_selection_logits, 0.0, 1.0)
    
    def initialize(self, input_sample: torch.Tensor):
        """Data-aware initialization from first batch."""
        # Sample feature values for threshold initialization
        with torch.no_grad():
            for tree_idx in range(self.num_trees):
                random_indices = torch.randint(0, input_sample.shape[0], (self.depth,))
                for depth_idx in range(self.depth):
                    sample_idx = random_indices[depth_idx]
                    # Initialize threshold with random feature value
                    feature_vals = input_sample[sample_idx]
                    self.feature_thresholds[tree_idx, depth_idx] = feature_vals.mean()
            
            # Initialize temperatures for linear region
            self.log_temperatures.data.fill_(0.0)  # exp(0) = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_features)
        Returns:
            (batch_size, num_trees * tree_dim) if flatten_output
            (batch_size, num_trees, tree_dim) otherwise
        """
        batch_size = x.shape[0]
        
        # Feature selection: (num_trees, depth, in_features) -> (num_trees, depth, batch_size)
        feature_weights = self.choice_function(
            self.feature_selection_logits, dim=-1
        )  # (num_trees, depth, in_features)
        
        # Selected features: (num_trees, depth, batch_size)
        feature_values = torch.einsum('tdf,bf->tdb', feature_weights, x)
        
        # Binary decisions: (num_trees, depth, batch_size)
        temperatures = torch.exp(self.log_temperatures).unsqueeze(-1)  # (num_trees, depth, 1)
        threshold_logits = (feature_values - self.feature_thresholds.unsqueeze(-1)) / temperatures
        
        # Bin probabilities: (num_trees, depth, batch_size)
        bin_probs = self.bin_function(threshold_logits)
        
        # Compute choice tensor via outer products
        # Start with (num_trees, batch_size, 1)
        choice = torch.ones(self.num_trees, batch_size, 1, device=x.device)
        
        for depth_idx in range(self.depth):
            # (num_trees, batch_size, 1)
            pi = bin_probs[:, depth_idx, :].unsqueeze(-1)
            # Outer product: concatenate [pi, 1-pi] and expand
            choice = torch.cat([
                choice * pi,
                choice * (1 - pi)
            ], dim=-1)
        
        # choice: (num_trees, batch_size, 2^depth)
        # response: (num_trees, 2^depth, tree_dim)
        # output: (num_trees, batch_size, tree_dim)
        output = torch.einsum('tbl,tld->tbd', choice, self.response)
        
        # Rearrange to (batch_size, num_trees, tree_dim)
        output = output.permute(1, 0, 2)
        
        if self.flatten_output:
            output = output.reshape(batch_size, -1)
        
        return output


class NeuralObliviousTreeWrapper:
    """Wrapper for GBDT integration - trains single tree on Newton targets."""
    
    def __init__(
        self,
        input_dim: int,
        num_trees: int = 64,
        depth: int = 5,
        tree_dim: int = 1,
        lr: float = 1e-2,
        epochs: int = 50,
        batch_size: int = 4096,
        temperature: float = 1.0,
        dropout_rate: float = 0.05,
        weight_decay: float = 5e-5,
        temp_min: float = 0.3,
        device: str = 'cuda',
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.temp_min = temp_min
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        self.odst = ODST(
            in_features=input_dim,
            num_trees=num_trees,
            depth=depth,
            tree_dim=tree_dim,
            flatten_output=True,
        ).to(self.device)
        
        self.head = nn.Linear(num_trees * tree_dim, 1).to(self.device)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self._initialized = False
        self._rescale = 1.0
    
    def fit(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
    ) -> None:
        """
        Fit on Newton targets: r = -grad / (hess + eps)
        with sample weights = hess
        """
        # Prepare data
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        g = torch.as_tensor(grad, dtype=torch.float64, device=self.device)
        h = torch.as_tensor(hess, dtype=torch.float64, device=self.device)
        
        # Newton targets
        r = -g / (h + 1e-8)
        r = r.to(torch.float32).view(-1, 1)
        w = torch.clamp(h, min=1e-12).to(torch.float32).view(-1, 1)
        
        # Data-aware initialization
        if not self._initialized:
            with torch.no_grad():
                sample_size = min(8192, X.shape[0])
                self.odst.initialize(X[:sample_size])
            self._initialized = True
        
        # Optimizer
        params = list(self.odst.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Training loop
        N = X.shape[0]
        for epoch in range(self.epochs):
            # Temperature annealing
            self._anneal_temperature(epoch)
            
            # Mini-batch SGD
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                idx = indices[start:start + self.batch_size]
                Xb, rb, wb = X[idx], r[idx], w[idx]
                
                optimizer.zero_grad()
                features = self.odst(Xb)
                features = self.dropout(features)
                pred = self.head(features)
                
                loss = torch.mean(wb * (pred - rb) ** 2)
                loss.backward()
                optimizer.step()
            
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                with torch.no_grad():
                    features = self.odst(X)
                    pred = self.head(features)
                    val_loss = torch.mean(w * (pred - r) ** 2).item()
                    print(f"[NeuralODT] epoch={epoch:03d} loss={val_loss:.6f}")
        
        # Calibration: scale outputs to reasonable magnitude
        with torch.no_grad():
            features = self.odst(X[:min(1000, N)])
            pred = self.head(features).cpu().numpy()
            med = np.median(np.abs(pred)) + 1e-12
            self._rescale = 1.0 / max(med, 1e-6)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        self.odst.eval()
        self.head.eval()
        
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            features = self.odst(X_t)
            pred = self.head(features).squeeze(1).cpu().numpy()
        
        return self._rescale * pred
    
    def _anneal_temperature(self, epoch: int):
        """Cosine annealing of temperature."""
        t0, t1 = self.temperature, self.temp_min
        cos_val = 0.5 * (1 + np.cos(np.pi * epoch / max(1, self.epochs)))
        new_temp = t1 + (t0 - t1) * cos_val
        
        with torch.no_grad():
            # Temperature is stored as log
            self.odst.log_temperatures.data.fill_(np.log(new_temp))


# ==================== Entmax implementations ====================

def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entmax with alpha=1.5 (entmax.entmax15 compatible)."""
    from entmax import entmax15 as _entmax15
    return _entmax15(logits, dim=dim)


def entmax15_2d(logits: torch.Tensor) -> torch.Tensor:
    """Two-class entmax for binary decisions."""
    stacked = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
    return entmax15(stacked, dim=-1)[..., 0]


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax transformation."""
    from entmax import sparsemax as _sparsemax
    return _sparsemax(logits, dim=dim)


def sparsemax_2d(logits: torch.Tensor) -> torch.Tensor:
    """Two-class sparsemax for binary decisions."""
    stacked = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
    return sparsemax(stacked, dim=-1)[..., 0]


# ==================== C++ interface functions ====================

def create_neural_tree(
    input_dim: int,
    config: Optional[dict] = None,
) -> NeuralObliviousTreeWrapper:
    """
    Factory function for C++ to create neural trees.
    
    Args:
        input_dim: Number of input features
        config: Optional dict with hyperparameters
    
    Returns:
        NeuralObliviousTreeWrapper instance
    """
    cfg = config or {}
    return NeuralObliviousTreeWrapper(
        input_dim=input_dim,
        num_trees=cfg.get('num_trees', 64),
        depth=cfg.get('depth', 5),
        tree_dim=cfg.get('tree_dim', 1),
        lr=cfg.get('lr', 1e-2),
        epochs=cfg.get('epochs', 50),
        batch_size=cfg.get('batch_size', 4096),
        temperature=cfg.get('temperature', 1.0),
        dropout_rate=cfg.get('dropout_rate', 0.05),
        weight_decay=cfg.get('weight_decay', 5e-5),
        device=cfg.get('device', 'cuda'),
        verbose=cfg.get('verbose', False),
    )


def fit_neural_tree(
    tree: NeuralObliviousTreeWrapper,
    X: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
) -> None:
    """Fit neural tree (called from C++)."""
    tree.fit(X, grad, hess)


def predict_neural_tree(
    tree: NeuralObliviousTreeWrapper,
    X: np.ndarray,
) -> np.ndarray:
    """Predict with neural tree (called from C++)."""
    return tree.predict(X)