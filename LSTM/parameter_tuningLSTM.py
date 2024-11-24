import os
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMAllocationModelWithAttention
from dataloader import StockDataLoader
from deepdow.losses import SharpeRatio, SortinoRatio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


##################### PARTS OF THIS CODE WAS WRITTEN WITH THE HELP OF GENERATIVE AI #####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
os.makedirs("models1", exist_ok=True)

def batch_entropy_penalty(allocations):
    batch_probabilities = allocations / allocations.sum(dim=1, keepdim=True)
    entropy = -torch.sum(batch_probabilities * torch.log(batch_probabilities + 1e-8), dim=1).mean()
    return -entropy



hidden_unit_options = [[148]]
sequence_length_options = [30]
learning_rate_list = [1e-3]
weight_decay = 1e-5
patience = 500
batch_size = 32
num_epochs = 1000
dropout_rate = 0.3
max_norm_list = [np.inf]
day_decay_list = [1]

asset_list = pd.read_csv('data/grouped_data_return_daily.csv').columns.tolist()[1:]
results = []
allocation_records = []
gradient_norms = []
#gamma = 1
for learning_rate in tqdm(learning_rate_list):
    for day_decay in day_decay_list:
        for max_norm in max_norm_list:
            for hidden_units in hidden_unit_options:
                for sequence_length in sequence_length_options:
                    wandb.init(
                        project="LSTMtest1",
                        config={
                            "num_layers": len(hidden_units),
                            "hidden_units": hidden_units,
                            "sequence_length": sequence_length,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "num_epochs": num_epochs,
                            "dropout_rate": dropout_rate,
                            "max_norm": max_norm,
                            "weight_decay": weight_decay,
                            "day_decay": day_decay
                        },
                        name=f"LSTM_{len(hidden_units)}_seq_{sequence_length}_hidden_{hidden_units}_lr_{learning_rate}_daydecay_{day_decay}_maxnorm_{max_norm}"
                    )

                    data_loader = StockDataLoader(
                        normalize_method="minmax",
                        csv_file='data/grouped_data_return_daily.csv',
                        sequence_length=sequence_length,
                        horizon=7,
                        batch_size=batch_size,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        top_n_symbols=148
                    )
                    train_loader = data_loader.get_train_loader()
                    val_loader = data_loader.get_val_loader()

                    config = {
                        "input_size": data_loader.train_data.shape[1],
                        "hidden_sizes": hidden_units,
                        "dropout_rate": dropout_rate,
                        "sequence_length": sequence_length,
                        "day_decay": day_decay
                    }

                    model = LSTMAllocationModelWithAttention(
                        input_size=config["input_size"], 
                        hidden_sizes=config["hidden_sizes"], 
                        dropout_rate=config["dropout_rate"], 
                        entmax_alpha=1.5,
                        decay_weight=config["day_decay"]
                    ).to(device)

                    loss = SharpeRatio() 
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    best_val_loss = float('inf')
                    epochs_no_improve = 0

                    allocation_epoch = []

                    for epoch in range(num_epochs):
                        epoch_gradient_norm = 0
                        model.train()
                        total_loss = 0

                        

                        for seq_x, seq_y in train_loader:
                            seq_x = seq_x.to(device)
                            seq_y = seq_y.to(device).unsqueeze(1)
                            

                            allocations = model(seq_x)
                            penalty = batch_entropy_penalty(allocations)
                            error = (loss(allocations, seq_y).mean() * np.sqrt(52)) #+ gamma * penalty
                            allocation_epoch.append(allocations.cpu().detach().numpy().mean(axis=0))
                            optimizer.zero_grad()
                            error.backward()

                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                            optimizer.step()

                            total_loss += error.item()

                            grad_norm = 0
                            for param in model.parameters():
                                if param.grad is not None:
                                    grad_norm += param.grad.data.norm(2).item() ** 2
                            grad_norm = grad_norm ** 0.5
                            epoch_gradient_norm += grad_norm



                        avg_train_loss = total_loss / len(train_loader)
                        avg_gradient_norm = epoch_gradient_norm / len(train_loader)

                        gradient_norms.append({
                            "Epoch": epoch,
                            "Gradient Norm": avg_gradient_norm
                        })

                        model.eval()
                        total_val_loss = 0
                        with torch.no_grad():
                            for seq_x, seq_y in val_loader:
                                seq_x = seq_x.to(device)
                                seq_y = seq_y.to(device).unsqueeze(1)
                                allocations = model(seq_x)
                                penalty = batch_entropy_penalty(allocations)
                                
                                val_error = (loss(allocations, seq_y).mean() * np.sqrt(52)) #+ gamma * penalty
                                total_val_loss += val_error.item()

                                allocation_records.append({
                                    "Epoch": epoch,
                                    "Allocations": allocations.tolist()
                                })

                        avg_val_loss = total_val_loss / len(val_loader)

                        wandb.log({
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "gradient_norm": avg_gradient_norm,
                            "epoch": epoch
                        })

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            epochs_no_improve = 0
                            best_model_filename = f"models1/5TESTALLbest_model_layers_{len(hidden_units)}_seq_{sequence_length}_hunits{hidden_units}_lr{learning_rate}_daydecay{day_decay}_maxnorm{max_norm}.pth"
                            torch.save({
                                "model_state_dict": model.state_dict(),
                                "config": config
                            }, best_model_filename)

                        else:
                            epochs_no_improve += 1

                        if epochs_no_improve >= patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            break

                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                print(f"Layer {name}, Grad Norm: {param.grad.norm().item()}")

                    checkpoint = torch.load(best_model_filename)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    sharpe_ratio = 1 / best_val_loss

                    final_model_filename = f"models1/5TESTALLfinal_model_layers_{len(hidden_units)}_seq_{sequence_length}_hunits{hidden_units[0]}_lr{learning_rate}_daydecay{day_decay}_maxnorm_{max_norm}.pth"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config
                    }, final_model_filename)

                    results.append({
                        "Model": f"Layers: {len(hidden_units)}, Hidden Units: {hidden_units}, Seq Len: {sequence_length}",
                        "Sharpe Ratio": sharpe_ratio,
                        "Best Validation Loss": best_val_loss
                    })

                    wandb.finish()

            results_df = pd.DataFrame(results)
            results_df.to_csv("test_hyperparameter_tuning_results.csv", index=False)

allocation_df = pd.DataFrame(allocation_records)
allocation_filename = "model-tests-discussion/allocations.csv"
allocation_df.to_csv(allocation_filename, index=False)

gradient_norm_df = pd.DataFrame(gradient_norms)
gradient_norm_filename = "model-tests-discussion/gradient_norms.csv"
gradient_norm_df.to_csv(gradient_norm_filename, index=False)

# Visualization of Allocations Heatmap
for epoch in allocation_df['Epoch'].unique():
    epoch_allocations = allocation_df[allocation_df['Epoch'] == epoch]
    
    # Convert list of allocations to a consistent 3D NumPy array
    allocation_list = epoch_allocations['Allocations'].tolist()
    allocation_array = np.array([np.mean(a, axis=0) if isinstance(a, np.ndarray) else np.mean(np.array(a), axis=0) for a in allocation_list])

    # Compute the mean across all batches for the epoch
    allocations_2d = np.mean(allocation_array, axis=0)
    allocations_2d = np.expand_dims(allocations_2d, axis=0)

    # Heatmap Visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(allocations_2d, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f"Allocations Heatmap - Epoch {epoch}")
    plt.xlabel("Assets")
    plt.ylabel("Sequence Positions")
    plt.show()
