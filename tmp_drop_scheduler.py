
def plot_dropout_behavior():
    set_seed(42)
    model = MLP(in_dim=2, out_dim=2, hidden_dim=16, depth=2, activation="relu", dropout=0.5)
    initialize_model(model, "kaiming")
    model.train()

    with torch.no_grad():
        x = Xc_train[:1].to(device)
        hidden = model.input(x)
        hidden = model.input_norm(hidden)
        hidden = model.act(hidden)

    sampled = []
    for i in range(6):
        dropped = model.blocks[0].dropout(hidden)
        sampled.append(dropped[0, :16].cpu().numpy())

    sampled = np.stack(sampled)

    plt.figure(figsize=(10, 4))
    for idx, values in enumerate(sampled, start=1):
        plt.plot(values, marker="o", label=f"pass {idx}")
    plt.title("Dropout on first-layer activations for one sample")
    plt.xlabel("Neuron index")
    plt.ylabel("Activation value")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.legend(loc="upper right")
    plt.show()

    zeros = (sampled == 0.0).sum(axis=0)
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(zeros)), zeros)
    plt.title("Dropout deactivation count across 6 forward passes")
    plt.xlabel("Neuron index")
    plt.ylabel("Zeroed passes")
    plt.show()


def run_dropout_experiment():
    histories = {}
    trained_models = {}
    configs = {
        "No Dropout": 0.0,
        "Dropout 0.3": 0.3,
        "Dropout 0.5": 0.5,
    }
    for label, p in configs.items():
        set_seed(42)
        model = MLP(
            in_dim=2,
            out_dim=2,
            hidden_dim=64,
            depth=4,
            activation="relu",
            dropout=p,
        )
        initialize_model(model, "kaiming")
        histories[label] = train_model(
            model,
            TensorDataset(Xc_train, yc_train),
            (Xc_test, yc_test),
            task="classification",
            optimizer_name="adam",
            lr=1e-3,
            batch_size=64,
            epochs=60,
        )
        trained_models[label] = model.to(device)
    return histories, trained_models


def train_with_scheduler(
    model,
    train_ds,
    test_data,
    task: str,
    optimizer_name: str,
    lr: float,
    scheduler_cls,
    scheduler_kwargs,
    batch_size: int = 64,
    epochs: int = 60,
):
    model = model.to(device)
    opt = make_optimizer(optimizer_name, model, lr=lr)
    scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls is not None else None
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    history = {
        "train_loss": [],
        "test_loss": [],
        "lr": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        te_loss, _ = evaluate_classification(model, *test_data) if task == "classification" else evaluate_regression(model, *test_data)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["test_loss"].append(te_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

    return history


def run_scheduler_experiment():
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

    histories = {}
    schedulers = {
        "Constant LR": (None, {}),
        "StepLR": (StepLR, {"step_size": 20, "gamma": 0.5}),
        "CosineAnnealingLR": (CosineAnnealingLR, {"T_max": 60}),
    }
    for label, (sched_cls, sched_kwargs) in schedulers.items():
        set_seed(42)
        model = MLP(
            in_dim=2,
            out_dim=2,
            hidden_dim=64,
            depth=4,
            activation="relu",
        )
        initialize_model(model, "kaiming")
        histories[label] = train_with_scheduler(
            model,
            TensorDataset(Xc_train, yc_train),
            (Xc_test, yc_test),
            task="classification",
            optimizer_name="adam",
            lr=1e-3,
            scheduler_cls=sched_cls,
            scheduler_kwargs=sched_kwargs,
            batch_size=64,
            epochs=60,
        )
    return histories
