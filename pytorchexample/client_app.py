import json
import random
from importlib import import_module

torch = import_module("torch")
import flwr.app
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: flwr.app.Message, context: flwr.app.Context):
    """Train the model on local data and maintain state across rounds."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function using the dynamic learning rate from the server
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # ---------------- STATEFUL CLIENT PERSISTENCE (Modern Flower 1.13+) ----------------
    if "fit_metrics" not in context.state:
        context.state["fit_metrics"] = flwr.app.MetricRecord({"training_loss_history": []})

    # Retrieve and append loss history
    loss_history = context.state["fit_metrics"].get("training_loss_history", [])
    loss_history.append(float(train_loss))
    context.state["fit_metrics"]["training_loss_history"] = loss_history
    print(f"Client {partition_id} Persistent Loss History: {loss_history}")
    # ----------------------------------------------------------------------------------

    # 1. Create a complex Python dictionary
    complex_metric = {
        "a": 1,
        "b": random.random(),
        "my_list": [1, 2, 3, 4]
    }
    
    # 2. Serialize to JSON, then convert string characters to a list of integers
    json_str = json.dumps(complex_metric)
    json_int_list = [ord(char) for char in json_str]

    # Construct and return reply Message
    model_record = flwr.app.ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "my_metric": json_int_list,
    }
    metric_record = flwr.app.MetricRecord(metrics)
    content = flwr.app.RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return flwr.app.Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: flwr.app.Message, context: flwr.app.Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "accuracy": eval_acc,  
        "num-examples": len(valloader.dataset),
    }
    metric_record = flwr.app.MetricRecord(metrics)
    content = flwr.app.RecordDict({"metrics": metric_record})
    
    return flwr.app.Message(content=content, reply_to=msg)
