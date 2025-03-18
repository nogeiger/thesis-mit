import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from models import NoisePredictorInitial, NoisePredictorLSTM, NoisePredictorTransformer, NoisePredictorGRU, NoisePredictorConv1D,NoisePredictorTCN
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, validate_model_diffusion, test_model, inference_application
from utils import loss_function, loss_function_start_point, add_noise, calculate_max_noise_factor
from datetime import datetime

def main():
    """
    Main function to execute the training and validation of the NoisePredictor model.
    """ 
    
    # Definition of parameters
    seq_length = 16 #seq len of data
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 2048#512#(Conv1D)#512(TCN)#256(Transformer#512(FF) #hidden dim of the model
    batch_size =64 #batch size
    num_epochs = 500#00#500#00 #number of epochs
    learning_rate = 1e-3 #learning rate
    noiseadding_steps = 20 # Number of steps to add noise
    use_forces = True  # Set this to True if you want to use forces as input to the model
    noise_with_force = False#True # Set this to True if you want to use forces as the noise
    #if force is used as noise, then force should not be used as input
    if noise_with_force:
            use_forces = False
    beta_start = 0.0001 #for the noise diffusion model
    beta_end = 0.02 #for the noise diffusion model
    max_grad_norm=7.0 #max grad norm for gradient clipping 
    add_gaussian_noise = False#True # to add additional guassian noise
    early_stop_patience = 25 #for early stopping
    save_interval = 20
    save_path = "save_checkpoints"
    timestamp = datetime.now().strftime("%Y-%"
    "m-%d_%H-%M-%S")

    hyperparams = {
    "seq_length": seq_length,
    "hidden_dim": hidden_dim,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "noiseadding_steps": noiseadding_steps,
    "use_forces": use_forces,
    "noise_with_force": noise_with_force,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "max_grad_norm": max_grad_norm,
    "add_gaussian_noise": add_gaussian_noise,
    "early_stop_patience": early_stop_patience
    }


    # File path to the real data
    file_path = "Data/RealData"

    # Load real data
    data = load_robot_data(file_path, seq_length, use_overlap=True)
    # Compute per-axis normalization statistics
    stats = compute_statistics_per_axis(data)
    # Normalize data per axis
    normalized_data = normalize_data_per_axis(data, stats)

    
    # Define split ratios
    train_ratio = 0.65 
    val_ratio = 0.2
    test_ratio = 0.1  

    # Compute split indices
    total_size = len(normalized_data)
    train_split = int(total_size * train_ratio)
    val_split = train_split + int(total_size * val_ratio)
    test_split = val_split + int(total_size * test_ratio)

    # Split data
    train_data = normalized_data[:train_split]
    val_data = normalized_data[train_split:val_split]
    test_data = normalized_data[val_split:test_split]
    application_data = normalized_data[test_split:]

    

    # Create datasets with per-axis normalization
    train_dataset = ImpedanceDatasetDiffusion(train_data, stats)
    val_dataset = ImpedanceDatasetDiffusion(val_data, stats)
    test_dataset = ImpedanceDatasetDiffusion(test_data, stats)
    application_dataset = ImpedanceDatasetDiffusion(application_data, stats)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    application_loader = DataLoader(application_dataset, batch_size=1, shuffle=False)
    print(f"Total sequences loaded: {len(test_dataset)} for training, {len(test_dataset)} for validation.")


    
    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Transformer"
    #model = NoisePredictorInitial(seq_length, hidden_dim, use_forces=use_forces).to(device) 
    model = NoisePredictorTransformer(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorTCN(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorGRU(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorConv1D(seq_length, hidden_dim, use_forces=use_forces).to(device)
    


    # Save hyperparameters
    save_path = os.path.join(save_path, f"{model_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "hyperparameters.txt"), "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")



    #choose optimizer
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #Choose loss
    #criterion = nn.MSELoss()
    #criterion = loss_function_start_point
    criterion=nn.SmoothL1Loss()
    print("_____________________________________________")
    print("-----Training and Validation-----")
    # Train and validate
    train_losses, val_loss = train_model_diffusion(
        model,
        train_loader, 
        val_loader,
        optimizer, 
        criterion, 
        device, 
        num_epochs, 
        noiseadding_steps, 
        beta_start, 
        beta_end, 
        use_forces,
        noise_with_force, 
        max_grad_norm,
        add_gaussian_noise,
        save_interval, 
        save_path,
        early_stop_patience)
    
    #plot train losses
    # Ensure x-axis matches the number of recorded epochs
    epochs_trained = len(train_losses)
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss with thicker lines
    plt.plot(range(1, epochs_trained + 1), train_losses, color='darkgreen',linewidth=3, label='Training Loss')
    plt.plot(range(1, epochs_trained + 1), val_loss, color='darkblue', linestyle='--', linewidth=3, label='Validation Loss')

    # Increase font sizes for better readability
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel('Loss (based on noise)', fontsize=16, fontweight='bold')
    plt.title('Training and validation loss', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)

    # Increase tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', linewidth=0.7)  # Optional: Improve readability with a grid
    # Save the plot in the save_path folder
    loss_plot_path = os.path.join(save_path, "training_validation_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')  # Save the plot as a high-resolution image
    # Display the plot
    plt.show()
    
    
    # Save test plots
    save_path_test = os.path.join(save_path, f"{model_name}_{timestamp}_test")
    os.makedirs(save_path_test, exist_ok=True)
    # Save test plots
    save_path_test_processed = os.path.join(save_path, f"{model_name}_{timestamp}_test_postprocessed")
    os.makedirs(save_path_test_processed, exist_ok=True)


    
    # Testing
    # Load best model
    best_model_path = os.path.join(save_path, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)
    print("_____________________________________________")
    print("-----Test model without postprocessing-----")
    test_model(model, val_loader, val_dataset, device, use_forces, save_path = save_path_test, num_denoising_steps=noiseadding_steps, num_samples=50, postprocessing=False)
    print("_____________________________________________")
    print("-----Test model with postprocessing-----")
    test_model(model, val_loader, val_dataset, device, use_forces, save_path = save_path_test_processed, num_denoising_steps=noiseadding_steps, num_samples=50, postprocessing=True)
    
    
    #Inference application
    save_path_application = os.path.join(save_path, f"{model_name}_{timestamp}_inference_application")
    os.makedirs(save_path_application, exist_ok=True)

    # Number of sequences to process (adjust as needed)
    num_application_sequences = 10
    print("_____________________________________________")
    print("-----Inference on application data-----")
    # Run inference on application data
    inference_application(
        model,
        application_loader,
        application_dataset,
        device,
        use_forces=use_forces,
        save_path=save_path_application,
        num_sequences=num_application_sequences,
        num_denoising_steps=noiseadding_steps,
        postprocessing=True  # Set to False if you don't want postprocessing
    )
    
    
if __name__ == "__main__":
    main()
