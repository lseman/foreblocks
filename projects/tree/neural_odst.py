# neural_odst.py - Python side neural oblivious decision tree

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ODST(nn.Module):
    """Neural Oblivious Decision Tree (NODE) — differentiable oblivious tree.

    Implements the Oblivious Decision Forest model from the NODE paper
    (Perekrestov et al., 2019), where each tree applies the same split
    decisions across all samples but with learned feature selection,
    thresholds, and response values.

    Architecture
    ------------
    For each sample x ∈ R^d and each tree t:

    1. **Feature selection**: Sparse weights over features via entmax/sparsemax/softmax
       w_{t,d} = choice_function(selection_logits_{t,d}) ∈ R^d, Σ w = 1
    2. **Feature projection**: v_{t,d,b} = Σ_j w_{t,d,j} · x_b[j]
    3. **Binarization**: Binary decisions via temperature-scaled thresholding
       z_{t,d,b} = 1 if v_{t,d,b} ≥ θ_{t,d}/τ_{t,d}, else 0
    4. **Path encoding**: Leaf index = Σ_d z_{t,d,b} · 2^d ∈ {0, ..., 2^D-1}
    5. **Response lookup**: output_{t,b} = response[t, leaf_index, :]

    Parameters
    ----------
    in_features : int
        Number of input features (dimensionality of x).
    num_trees : int
        Number of oblivious trees in the ensemble. Default 64.
    depth : int
        Depth of each tree (number of binary splits). Each tree has 2^depth leaves.
            Default 6.
    tree_dim : int
        Output dimension per tree. For regression, use 1.
            Default 1.
    flatten_output : bool
        If True, concatenate all tree outputs into a single vector.
            If False, keep (batch, num_trees, tree_dim) shape. Default True.
    choice_function : str
        Feature selection function: 'entmax15', 'sparsemax', or 'softmax'.
            Default 'entmax15'.
    bin_function : str
        Binarization function: 'entmax15', 'sparsemax', or 'softmax' (binary).
            Default 'entmax15'.

    Attributes
    ----------
    feature_selection_logits : Tensor[num_trees, depth, in_features]
        Learnable logits for feature selection per tree/depth.
    feature_thresholds : Tensor[num_trees, depth]
        Learnable split thresholds per tree/depth.
    log_temperatures : Tensor[num_trees, depth]
        Log-temperature for binarization (annealed during training).
    response : Tensor[num_trees, 2^depth, tree_dim]
        Leaf response values — the "weights" of each leaf in each tree.

    Examples
    --------
    >>> import torch
    >>> from neural_odst import ODST
    >>> model = ODST(in_features=16, num_trees=64, depth=6)
    >>> x = torch.randn(32, 16)
    >>> y = model(x)  # (32, 64) — flattened output
    >>> assert y.shape == (32, 64)
    """
    
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
        """Get the binarization function by name.

        Parameters
        ----------
        name : str
            Function name: 'entmax15', 'sparsemax', or 'softmax'.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Bin function that takes logits and returns binary probabilities.
        """
        if name == 'entmax15':
            return entmax15_2d
        elif name == 'sparsemax':
            return sparsemax_2d
        else:  # softmax
            return lambda x: torch.softmax(torch.stack([x, torch.zeros_like(x)], -1), -1)[..., 0]

    def _get_choice_fn(self, name: str):
        """Get the feature selection function by name.

        Parameters
        ----------
        name : str
            Function name: 'entmax15', 'sparsemax', or 'softmax'.

        Returns
        -------
        Callable[[torch.Tensor, int], torch.Tensor]
            Choice function that takes logits and dim, returns sparse weights.
        """
        if name == 'entmax15':
            return entmax15
        elif name == 'sparsemax':
            return sparsemax
        else:
            return torch.softmax

    def _initialize_params(self):
        """Initialize feature selection logits uniformly.

        Data-aware threshold initialization is performed in the initialize()
        method using a sample of training data.
        """
        nn.init.uniform_(self.feature_selection_logits, 0.0, 1.0)

    def initialize(self, input_sample: torch.Tensor):
        """Data-aware initialization of thresholds from a sample batch.

        Samples random feature values from the input to set initial
        thresholds near the data distribution, improving convergence.

        Parameters
        ----------
        input_sample : torch.Tensor
            A representative batch of shape (sample_size, in_features).
                Typically the first training batch (up to 8192 samples).

        Examples
        --------
        >>> model = ODST(in_features=16)
        >>> sample = torch.randn(1024, 16)
        >>> model.initialize(sample)  # Initialize thresholds from data
        """
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
        """Forward pass through the oblivious decision forest.

        For each sample, computes feature selection weights, binarizes
        features at each depth level via temperature-scaled thresholds,
        traces the path to a leaf, and looks up the response value.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Predictions:
              - If flatten_output=True: (batch_size, num_trees * tree_dim)
              - If flatten_output=False: (batch_size, num_trees, tree_dim)

        Examples
        --------
        >>> model = ODST(in_features=16, num_trees=64, depth=6)
        >>> x = torch.randn(32, 16)
        >>> y = model(x)
        >>> assert y.shape == (32, 64)  # flatten_output=True by default
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
    """GBDT-style wrapper for NeuralODST — trains one tree per boosting round.

    Fits an ODST model on Newton's method targets (r = -grad / hess) with
    sample weights equal to the Hessian. Uses a two-stage architecture:

    1. **ODST backbone**: Learns feature selection, thresholds, and responses
       via differentiable oblivious tree forward pass.
    2. **Linear head**: Maps ODST output to scalar prediction.

    Temperature annealing (cosine schedule) sharpens binarization during training,
    driving the model toward hard binary decisions while maintaining gradient flow.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_trees : int
        Number of oblivious trees. Default 64.
    depth : int
        Tree depth (number of splits). Default 5.
    tree_dim : int
        Output dimension per tree. Default 1.
    lr : float
        Learning rate for AdamW optimizer. Default 1e-2.
    epochs : int
        Number of training epochs. Default 50.
    batch_size : int
        Mini-batch size for SGD. Default 4096.
    temperature : float
        Initial binarization temperature (τ₀). Higher = softer decisions.
            Default 1.0.
    temp_min : float
        Minimum temperature after cosine annealing (τ₁). Default 0.3.
    dropout_rate : float
        Dropout rate between ODST output and linear head. 0.0 disables.
            Default 0.05.
    weight_decay : float
        L2 regularization for AdamW. Default 5e-5.
    device : str
        PyTorch device ('cpu', 'cuda', etc.). Default auto-detects GPU.
    verbose : bool
        Print loss every 10 epochs. Default False.

    Attributes
    ----------
    odst : ODST
        The oblivious decision tree model.
    head : nn.Linear
        Linear layer mapping ODST output to scalar prediction.
    dropout : nn.Module
        Dropout layer (or Identity if dropout_rate=0).
    _rescale : float
        Output calibration factor (applied in predict()).

    Examples
    --------
    >>> import numpy as np
    >>> from neural_odst import NeuralObliviousTreeWrapper
    >>> wrapper = NeuralObliviousTreeWrapper(input_dim=16, num_trees=64, depth=5)
    >>> X = np.random.randn(4096, 16).astype(np.float32)
    >>> grad = np.random.randn(4096).astype(np.float64)
    >>> hess = np.ones(4096, dtype=np.float64) + 0.1
    >>> wrapper.fit(X, grad, hess)
    >>> pred = wrapper.predict(X[:10])
    """
    
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
        """Fit the neural oblivious tree on Newton's method targets.

        Computes residuals r = -grad / (hess + ε) and uses Hessian values
        as sample weights. Performs data-aware initialization from the first
        batch, then trains with AdamW and cosine temperature annealing.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, input_dim), float32.
        grad : np.ndarray
            Gradient vector of shape (n_samples,), float64.
        hess : np.ndarray
            Hessian vector of shape (n_samples,), float64. Used as sample weights.

        Notes
        -----
        Data-aware initialization is performed once on the first call using
        up to 8192 samples from X. Subsequent calls skip re-initialization.
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
        """Predict on new data with output calibration.

        Runs inference in eval mode and applies the learned rescale factor
        to produce predictions in a reasonable magnitude range.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, input_dim), float32 or float64.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,), float64.
        """
        self.odst.eval()
        self.head.eval()

        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            features = self.odst(X_t)
            pred = self.head(features).squeeze(1).cpu().numpy()

        return self._rescale * pred
        self.odst.eval()
        self.head.eval()
        
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            features = self.odst(X_t)
            pred = self.head(features).squeeze(1).cpu().numpy()
        
        return self._rescale * pred
    
    def _anneal_temperature(self, epoch: int):
        """Cosine anneal binarization temperature from τ₀ to τ₁.

        Temperature controls the sharpness of binary decisions. Higher
        temperature produces softer (more differentiable) decisions;
        lower temperature approaches hard 0/1 splits.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        """
        t0, t1 = self.temperature, self.temp_min
        cos_val = 0.5 * (1 + np.cos(np.pi * epoch / max(1, self.epochs)))
        new_temp = t1 + (t0 - t1) * cos_val

        with torch.no_grad():
            # Temperature is stored as log
            self.odst.log_temperatures.data.fill_(np.log(new_temp))
        t0, t1 = self.temperature, self.temp_min
        cos_val = 0.5 * (1 + np.cos(np.pi * epoch / max(1, self.epochs)))
        new_temp = t1 + (t0 - t1) * cos_val
        
        with torch.no_grad():
            # Temperature is stored as log
            self.odst.log_temperatures.data.fill_(np.log(new_temp))


# ==================== Entmax implementations ====================

def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entmax with α=1.5 (sparse, differentiable alternative to softmax).

    Entmax with α=1.5 produces sparse probability distributions — some
    elements can be exactly zero, unlike softmax. This encourages the model
    to select only a few features per split node.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of any shape.
    dim : int
        Dimension along which to apply entmax. Default -1.

    Returns
    -------
    torch.Tensor
        Sparse probability distribution with the same shape as *logits*.
    """
    from entmax import entmax15 as _entmax15
    return _entmax15(logits, dim=dim)


def entmax15_2d(logits: torch.Tensor) -> torch.Tensor:
    """Two-class entmax for binary binarization decisions.

    Stacks logits alongside zeros and applies entmax15, returning the
    probability of the "left" (≤ threshold) branch.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of shape (*, d).

    Returns
    -------
    torch.Tensor
        Probability of the left branch, shape (*, d).
    """
    stacked = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
    return entmax15(stacked, dim=-1)[..., 0]


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute sparsemax (α=2, sparse alternative to softmax).

    Sparsemax produces exactly zero probabilities for some elements,
    enabling feature selection in the ODST model.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of any shape.
    dim : int
        Dimension along which to apply sparsemax. Default -1.

    Returns
    -------
    torch.Tensor
        Sparse probability distribution with the same shape as *logits*.
    """
    from entmax import sparsemax as _sparsemax
    return _sparsemax(logits, dim=dim)


def sparsemax_2d(logits: torch.Tensor) -> torch.Tensor:
    """Two-class sparsemax for binary binarization decisions.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of shape (*, d).

    Returns
    -------
    torch.Tensor
        Probability of the left branch, shape (*, d).
    """
    stacked = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
    return sparsemax(stacked, dim=-1)[..., 0]


# ==================== C++ interface functions ====================

def create_neural_tree(
    input_dim: int,
    config: Optional[dict] = None,
) -> NeuralObliviousTreeWrapper:
    """Factory function to create a NeuralObliviousTreeWrapper.

    Used by C++ code (via pybind11/nanobind) to instantiate Python-side
    neural trees. Also usable directly from Python.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    config : dict | None
        Optional hyperparameter overrides:

        - num_trees (int): Default 64
        - depth (int): Default 5
        - tree_dim (int): Default 1
        - lr (float): Default 1e-2
        - epochs (int): Default 50
        - batch_size (int): Default 4096
        - temperature (float): Default 1.0
        - dropout_rate (float): Default 0.05
        - weight_decay (float): Default 5e-5
        - device (str): Default 'cuda'
        - verbose (bool): Default False

    Returns
    -------
    NeuralObliviousTreeWrapper
        Configured wrapper instance ready for fit().

    Examples
    --------
    >>> tree = create_neural_tree(input_dim=16, config={'num_trees': 32})
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
    """Fit a neural oblivious tree on Newton's method targets.

    Thin wrapper around NeuralObliviousTreeWrapper.fit() for use by C++
    code via nanobind bindings.

    Parameters
    ----------
    tree : NeuralObliviousTreeWrapper
        Pre-created tree instance (via create_neural_tree).
    X : np.ndarray
        Feature matrix (n_samples, input_dim), float32.
    grad : np.ndarray
        Gradient vector (n_samples,), float64.
    hess : np.ndarray
        Hessian vector (n_samples,), float64.
    """
    tree.fit(X, grad, hess)


def predict_neural_tree(
    tree: NeuralObliviousTreeWrapper,
    X: np.ndarray,
) -> np.ndarray:
    """Predict with a fitted neural oblivious tree.

    Thin wrapper around NeuralObliviousTreeWrapper.predict() for use by C++
    code via nanobind bindings.

    Parameters
    ----------
    tree : NeuralObliviousTreeWrapper
        Fitted tree instance.
    X : np.ndarray
        Feature matrix (n_samples, input_dim), float32 or float64.

    Returns
    -------
    np.ndarray
        Predictions (n_samples,), float64.
    """
    return tree.predict(X)