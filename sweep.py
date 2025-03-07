import wandb
import argparse
from train import get_args, train_model

# Define sweep configuration
sweep_config = {
    "method": "bayes",  # Bayesian Optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]}
    }
}

# Step 1: Create the sweep
sweep_id = wandb.sweep(sweep_config, project="myprojectname")#DA6401-A1
print(f"Created new sweep with ID: {sweep_id}")

def sweep_train():
    # Get default args (Namespace object)
    default_config = get_args()

    # Initialize WandB
    wandb.init(project=default_config.wandb_project)

    # Override default args with sweep-configured values
    sweep_params = wandb.config
    for param, value in sweep_params.items():
        setattr(default_config, param, value)

    # Create a meaningful run name
    run_name = (f"nl_{default_config.num_layers}_bs_{default_config.batch_size}_hs_{default_config.hidden_size}_"
                f"ac_{default_config.activation}")
    
    wandb.run.name = run_name  # Set custom run name
    wandb.run.save()

    # Train model
    train_model(default_config)

# Step 2: Run the sweep
wandb.agent(sweep_id, function=sweep_train, count=50)  # Run 50 trials


# Step 3: Retrieve the best run **AFTER** all trials are completed
entity = "myname"
project = "myprojectname"
api = wandb.Api()

try:
    # Fetch the sweep results
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Sort runs by validation accuracy
    runs = sorted(sweep.runs, key=lambda run: run.summary.get("Val Accuracy", 0), reverse=True)

    if runs:
        best_run = runs[0]  # Best run with highest validation accuracy
        val_accuracy = best_run.summary.get("Val Accuracy", 0)
        best_params = best_run.config

        print(f"Best run: {best_run.name} with {val_accuracy} accuracy")
        print("Best hyperparameters:", best_params)
    else:
        print("No runs found in the sweep.")
except wandb.errors.CommError:
    print("Error: Sweep not found. Ensure the sweep ID is correct and exists.")
