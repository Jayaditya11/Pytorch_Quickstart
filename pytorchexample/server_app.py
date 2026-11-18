"""pytorchexample: A Flower / PyTorch app."""

import json
import os
from datetime import datetime
import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

def weighted_average(reply_contents: list[RecordDict], weight_key: str) -> MetricRecord:
    """Aggregate distributed evaluation accuracies using modern RecordDicts."""
    accuracies = []
    examples = []
    for content in reply_contents:
        m = content["metrics"]
        num_ex = m[weight_key]  # automatically defaults to "num-examples"
        accuracies.append(num_ex * m["accuracy"])
        examples.append(num_ex)
    return MetricRecord({"accuracy": sum(accuracies) / sum(examples)})

def handle_fit_metrics(reply_contents: list[RecordDict], weight_key: str) -> MetricRecord:
    """Process and aggregate complex fit metrics from clients."""
    b_values = []
    for content in reply_contents:
        # Extract the JSON int list from the modern MetricRecord
        my_metric_list = content["metrics"]["my_metric"]
        my_metric_str = "".join([chr(c) for c in my_metric_list])
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return MetricRecord({"max_b": max(b_values)})

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    wandb.init(
        project="flower-federated-learning",
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy purely with modern callbacks
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=weighted_average,
        train_metrics_aggr_fn=handle_fit_metrics, 
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}), # Constant LR is natively broadcast here
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
        
    wandb.finish()

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data and execute custom logic."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    
    # Save checkpoint per round
    torch.save(model.state_dict(), f"global_model_round_{server_round}.pt")
    
    # Save metrics to local JSON
    results_file = "metrics.json"
    history = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            history = json.load(f)
            
    history[str(server_round)] = {"loss": test_loss, "accuracy": test_acc}
    with open(results_file, "w") as f:
        json.dump(history, f, indent=4)
        
    # Log to Weights & Biases
    wandb.log({"centralized_loss": test_loss, "centralized_accuracy": test_acc}, step=server_round)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})