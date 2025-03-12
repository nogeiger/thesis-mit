import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
from utils import loss_function, add_noise
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from datetime import datetime



def train_model_diffusion(model, traindataloader, valdataloader,optimizer, criterion, device, num_epochs, noiseadding_steps, beta_start, 
                          beta_end, use_forces=False, noise_with_force=False, max_grad_norm=7.0, add_gaussian_noise=False, save_interval = 20, 
                          save_path = "save_checkpoints",early_stop_patience = 25):
    """
    Trains the NoisePredictor model using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for training (CPU or GPU).
        num_epochs (int): Number of training epochs.
        noiseadding_steps (int): Number of steps to add noise.
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        list: List of average losses for each epoch.
    """
    train_epoch_losses = []
    val_epoch_losses = []

    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    best_val_loss = float('inf')  # Track best validation loss
    early_stopping_counter = 0  # Count epochs since last improvement


    # Initialize ReduceLROnPlateau
    lr_scheduler_patience = max(5, int(early_stop_patience * 0.32))  # Reduce LR after 1/3 of early stopping patience
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm to create a progress bar
        for batch_idx, (pos_0, pos, u_0, u, theta_0, theta, force, moment) in enumerate(tqdm(traindataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)):
            
            # Move data to device
            clean_pos = pos_0.to(device)
            complete_noisy_pos = pos.to(device)
            clean_u = u_0.to(device)
            complete_noisy_u = u.to(device)
            clean_theta = theta_0.to(device)
            complete_noisy_theta = theta.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_u, ground_truth_alpha, noisy_theta, noise_scale = add_noise(clean_pos, complete_noisy_pos, 
                                        clean_u, complete_noisy_u, clean_theta, complete_noisy_theta,
                                        force, moment,
                                        noiseadding_steps, beta_start, 
                                        beta_end, noise_with_force, add_gaussian_noise)
            
            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default noise


            actual_noise_theta = noisy_theta - clean_theta

            optimizer.zero_grad()

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_u, noisy_theta, force, moment)
            
            else:
                predicted_noise = model(noisy_pos, noisy_u, noisy_theta)
            
            # Calculate loss and perform backward pass
            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) + criterion(predicted_noise[:,:,-2], ground_truth_alpha) + criterion(predicted_noise[:,:,-1], actual_noise_theta.squeeze(-1)) 
            loss = loss / torch.clamp(noise_scale, min=1e-6) * 10000  # Normalize loss by noise scale
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(traindataloader)
        train_epoch_losses.append(avg_train_loss)


        #Validation after each epoch
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_loss = validate_model_diffusion(
                model, valdataloader, criterion, device, noiseadding_steps, beta_start, beta_end, 
                use_forces, noise_with_force, add_gaussian_noise
            )
        val_epoch_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        last_lr = optimizer.param_groups[0]['lr']
        # Reduce LR if no improvement for `lr_scheduler_patience` epochs
        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"Epoch {epoch+1}: Learning Rate dropped to = {current_lr:.6e}")
  

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} after epoch {epoch+1}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping patience: {early_stopping_counter}/{early_stop_patience}")

        # If no improvement for `patience` epochs, stop training
        if early_stopping_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Restoring best model.")
            model.load_state_dict(torch.load(best_model_path))  # Restore best model
            break  # Exit training loop

        # Save model every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    

    return train_epoch_losses, val_epoch_losses


def validate_model_diffusion(model, dataloader, criterion, device, max_noiseadding_steps, 
                             beta_start, beta_end, use_forces=False, noise_with_force=False, add_gaussian_noise=False):
    """
    Validates the NoisePredictor model on unseen data using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for validation (CPU or GPU).
        max_noiseadding_steps (int): Maximum number of steps to add noise.
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0

    # Use tqdm to create a progress bar
    with torch.no_grad():
        for batch_idx, (pos_0, pos, u_0, u, theta_0, theta, force, moment)in enumerate(tqdm(dataloader, desc="Validating", leave=True)):
            clean_pos = pos_0.to(device)
            noisy_pos = pos.to(device)
            clean_u = u_0.to(device)
            noisy_u = u.to(device)
            clean_theta = theta_0.to(device)
            noisy_theta = theta.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_u, ground_truth_alpha, noisy_theta, noise_scale = add_noise(clean_pos, noisy_pos, clean_u, noisy_u, clean_theta, noisy_theta,
                                        force, moment,
                                        max_noiseadding_steps, beta_start, beta_end, noise_with_force, add_gaussian_noise)

            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default: noise is the diff


            actual_noise_theta = noisy_theta - clean_theta

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_u, noisy_theta, force, moment)
            else:
                predicted_noise = model(noisy_pos, noisy_u, noisy_theta)

            # Calculate loss
            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) + criterion(predicted_noise[:,:,-2], ground_truth_alpha) + criterion(predicted_noise[:,:,-1], actual_noise_theta.squeeze(-1)) 
            loss = loss / torch.clamp(noise_scale, min=1e-6) * 10000
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, val_loader, val_dataset, device, use_forces, save_path, num_denoising_steps=1, num_samples=5, postprocessing = False):
    """
    Function to evaluate the model by predicting noise, performing iterative denoising,
    and visualizing the results.

    Args:
        model (torch.nn.Module): The trained noise predictor model.
        val_loader (DataLoader): DataLoader for the validation set.
        val_dataset (Dataset): Validation dataset (for denormalization).
        device (torch.device): Device (CPU/GPU).
        use_forces (bool): Whether forces are used as an input to the model.
        num_denoising_steps (int): Number of denoising steps.
        num_samples (int): Number of samples to visualize.
    """

    model.eval()  # Set model to evaluation mode

    # Convert validation dataset into a list (ensures correct indexing)
    val_data = list(val_loader.dataset)

    # Randomly select `num_samples` different indices from the validation dataset
    sample_indices = random.sample(range(len(val_data)), num_samples)

    # Initialize lists for mean absolute differences
    mean_diffs_pos_x, mean_diffs_pos_y, mean_diffs_pos_z, overall_mean_diffs_pos = [], [], [], []
    mean_diffs_u_x, mean_diffs_u_y, mean_diffs_u_z, overall_mean_diffs_u = [], [], [], []
    mean_diffs_theta = []


    for sample_idx, idx in enumerate(sample_indices):
        # Fetch a **random sample** instead of always using the first batch
        clean_pos, noisy_pos, clean_u, noisy_u, clean_theta, noisy_theta, force, moment = val_data[idx]

        # Move data to the correct device
        clean_pos = clean_pos.unsqueeze(0).to(device)  # Add batch dimension
        noisy_pos = noisy_pos.unsqueeze(0).to(device)
        clean_u = clean_u.unsqueeze(0).to(device)
        noisy_u = noisy_u.unsqueeze(0).to(device)
        clean_theta = clean_theta.unsqueeze(0).to(device)
        noisy_theta = noisy_theta.unsqueeze(0).to(device)
        force = force.unsqueeze(0).to(device)
        moment = moment.unsqueeze(0).to(device)

        # Start iterative denoising
        denoised_pos = noisy_pos.clone()
        denoised_u = noisy_u.clone()
        denoised_theta = noisy_theta.clone()
        for _ in range(num_denoising_steps):
            predicted_noise = model(denoised_pos, denoised_u, denoised_theta, force, moment) if use_forces else model(denoised_pos, denoised_u, denoised_theta)


            denoised_pos = denoised_pos - predicted_noise[:,:,0:3]  # Remove noise iteratively
            denoised_theta = denoised_theta - predicted_noise[:,:,-1].unsqueeze(-1)

            # Extract the predicted alpha (rotation noise)
            predicted_alpha = predicted_noise[:, :, -2]  # Shape: [batch, seq_length]

            # Choose a reference vector r that is not parallel to u_noisy
            r = torch.tensor([1.0, 0.0, 0.0], device=noisy_u.device).expand_as(noisy_u)
            r = torch.where(torch.abs(noisy_u) < 0.9, r, torch.tensor([0.0, 1.0, 0.0], device=noisy_u.device).expand_as(noisy_u))

            # Compute perpendicular component v
            v = r - (torch.sum(r * noisy_u, dim=-1, keepdim=True) * noisy_u)
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)  # Normalize v

            # Compute denoised_u using inverse rotation formula
            denoised_u = torch.cos(predicted_alpha).unsqueeze(-1) * noisy_u - torch.sin(predicted_alpha).unsqueeze(-1) * v



    
        # Denormalize trajectories
        noisy_pos_np = val_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
        clean_pos_np = val_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
        denoised_pos_np = val_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()

        noisy_u_np = val_dataset.denormalize(noisy_u.detach().cpu(), "u").numpy()
        clean_u_np = val_dataset.denormalize(clean_u.detach().cpu(), "u_0").numpy()
        denoised_u_np = val_dataset.denormalize(denoised_u.detach().cpu(), "u_0").numpy()

        noisy_theta_np = val_dataset.denormalize(noisy_theta.detach().cpu(), "theta").numpy()
        clean_theta_np = val_dataset.denormalize(clean_theta.detach().cpu(), "theta_0").numpy()
        denoised_theta_np = val_dataset.denormalize(denoised_theta.detach().cpu(), "theta_0").numpy()

        force_np = val_dataset.denormalize(force.detach().cpu(), "force").numpy()
        moment_np = val_dataset.denormalize(moment.detach().cpu(), "force").numpy()

        if postprocessing == True:
            #Preprocessing of denoise
            # Compute the offset using the first point difference
            #offset = clean_pos_np[:, 0, :] - denoised_pos_np[:, 0, :]
            # Apply the offset to all points in the denoised pos
            #denoised_pos_np += offset[:, np.newaxis, :]


            # Apply smoothing using a moving average filter
            window_size = 20  # Adjust the window size based on smoothing needs
            denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
            denoised_theta_np = uniform_filter1d(denoised_theta_np, size=window_size, axis=1, mode='nearest')
            #TO DO: Think about smooting and offset of u
            #denoised_u_np = uniform_filter1d(denoised_u_np, size=window_size, axis=1, mode='nearest')



            # Compute the offset using the average of the first 5 points difference
            offset_pos = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
            offset_theta = np.mean(clean_theta_np[:, :1, :] - denoised_theta_np[:, :1, :], axis=1)
            #remove offset from denoised u
            # Apply the offset to all points in the denoised pos
            denoised_pos_np += offset_pos[:, np.newaxis, :]
            denoised_theta_np += offset_theta[:, np.newaxis, :]
 
            # Apply rotation correction for offset of u_np --> rotation matrix needed for unit vectors
            # Compute cosine similarity (dot product)
            cos_alpha_offset = np.sum(clean_u_np[:, 0, :] * denoised_u_np[:, 0, :], axis=-1)
            cos_alpha_offset = np.clip(cos_alpha_offset, -1.0, 1.0)  # Ensure valid arccos range

            # Compute the rotation angle
            alpha_offset = np.arccos(cos_alpha_offset)  # Shape: [batch_size]

            # Compute perpendicular rotation axis v using cross product
            v = np.cross(denoised_u_np[:, 0, :], clean_u_np[:, 0, :])

            # Normalize v to unit length
            v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero

            # Apply rotation correction to all time steps
            denoised_u_np = np.cos(alpha_offset)[:, np.newaxis, np.newaxis] * denoised_u_np - np.sin(alpha_offset)[:, np.newaxis, np.newaxis] * v


            print("denoised_u", denoised_u)
            print("clean_u", clean_u_np)


        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
        overall_mean_diff_pos = np.mean(np.abs(clean_pos_np - denoised_pos_np))

        mean_diff_ux = np.mean(np.abs(clean_u_np[:, :, 0] - denoised_u_np[:, :, 0]))
        mean_diff_uy = np.mean(np.abs(clean_u_np[:, :, 1] - denoised_u_np[:, :, 1]))
        mean_diff_uz = np.mean(np.abs(clean_u_np[:, :, 2] - denoised_u_np[:, :, 2]))
        overall_mean_diff_u = np.mean(np.abs(clean_u_np - denoised_u_np))

        mean_diff_theta = np.mean(np.abs(clean_theta_np - denoised_theta_np))


        # Append mean differences to lists
        mean_diffs_pos_x.append(mean_diff_x)
        mean_diffs_pos_y.append(mean_diff_y)
        mean_diffs_pos_z.append(mean_diff_z)
        overall_mean_diffs_pos.append(overall_mean_diff_pos)

        mean_diffs_u_x.append(mean_diff_ux)
        mean_diffs_u_y.append(mean_diff_uy)
        mean_diffs_u_z.append(mean_diff_uz)
        overall_mean_diffs_u.append(overall_mean_diff_u)

        mean_diffs_theta.append(mean_diff_theta)


        # Create a separate figure for each sample
        fig, ax_traj = plt.subplots(1, 1, figsize=(12, 6))  # Wider figure for better visibility

        # Plot clean vs denoised pos (Y-axis only) with thicker lines
        ax_traj.plot(clean_u_np[0, :, 0], label='Clean (ground truth) ux', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_u_np[0, :, 0], linestyle='--', label='Denoised (diffusion model) ux', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_u_np[0, :, 1], label='Clean (ground truth) uy', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_u_np[0, :, 1], linestyle='--', label='Denoised (diffusion model) uy', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_u_np[0, :, 2], label='Clean (ground truth) uz', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_u_np[0, :, 2], linestyle='--', label='Denoised (diffusion model) uz', linewidth=3.5, color='darkgreen')

   
        # Customize plot appearance with bold labels and increased font size
        ax_traj.set_xlabel('Time Step', fontsize=16, fontweight='bold')
        ax_traj.set_ylabel(r'$\tilde{y}_o$ Position', fontsize=16, fontweight='bold')  # Y-label with tilde notation
        ax_traj.set_title(f'Clean vs denoised zero force pos in y-direction - Sample {sample_idx+1}', 
                        fontsize=18, fontweight='bold')

        ax_traj.legend(fontsize=14)

        # Make grid lines more visible
        ax_traj.grid(True, linestyle="--", linewidth=1, alpha=0.7)

        # Increase tick label size and make ticks thicker
        ax_traj.tick_params(axis='both', labelsize=14, width=2.5, length=8)

        # Define save path for the plot
        plot_filename = os.path.join(save_path, f"pos_sample_{sample_idx+1}.png")

        # Save the figure
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

        # Show plots without blocking execution
        #plt.show(block=False)

        # Close the figure to free memory
        plt.close(fig)


    # Define the path for the results file
    results_file = os.path.join(save_path, "test_results.txt")

    # Prepare the results text
    results_text = (
        f"\nMean Absolute Differences Across {num_samples} Samples:\n"
        f"X-axis: {np.mean(mean_diffs_pos_x):.6f}\n"
        f"Y-axis: {np.mean(mean_diffs_pos_y):.6f}\n"
        f"Z-axis: {np.mean(mean_diffs_pos_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs_pos):.6f}\n\n"
        f"U_x: {np.mean(mean_diffs_u_x):.6f}\n"
        f"U_y: {np.mean(mean_diffs_u_y):.6f}\n"
        f"U_z: {np.mean(mean_diffs_u_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs_u):.6f}\n\n"
        f"Theta: {np.mean(mean_diffs_theta):.6f}\n"
    )

    # Print results to console
    print(results_text)

    # Save results to file
    with open(results_file, "w") as file:
        file.write(results_text)

    print(f"Test results saved to {results_file}")


    # Keep plots open until the user closes them
    #plt.pause(0.1)
    #input("Press Enter to close all plots and continue...")
    plt.close('all')


def inference_application(model, application_loader, application_dataset, device, use_forces, save_path, num_sequences=100, num_denoising_steps=1, postprocessing=False):
    """
    Function to perform inference on the application dataset, reconstructing sequences sequentially.

    Args:
        model (torch.nn.Module): Trained noise predictor model.
        application_loader (DataLoader): DataLoader for the application dataset.
        application_dataset (Dataset): Application dataset (for denormalization).
        device (torch.device): Device (CPU/GPU).
        use_forces (bool): Whether forces are used as an input to the model.
        num_sequences (int): Number of sequences to process.
        num_denoising_steps (int): Number of denoising steps.
        postprocessing (bool): Whether to apply postprocessing smoothing.
    """
    
    model.eval()  # Set model to evaluation mode

    # Convert application dataset into a list for sequential access
    application_data = list(application_loader.dataset)
    
    # Ensure num_sequences does not exceed available data
    num_sequences = min(num_sequences, len(application_data))
    
    # Initialize lists for mean absolute differences
    mean_diffs_x, mean_diffs_y, mean_diffs_z, overall_mean_diffs = [], [], [], []

    for seq_idx in range(num_sequences):
        # Fetch the sequence in order
        clean_pos, noisy_pos, force = application_data[seq_idx]

        # Move data to the correct device
        clean_pos = clean_pos.unsqueeze(0).to(device)  # Add batch dimension
        noisy_pos = noisy_pos.unsqueeze(0).to(device)
        force = force.unsqueeze(0).to(device)

        # Start iterative denoising
        denoised_pos = noisy_pos.clone()
        for _ in range(num_denoising_steps):
            predicted_noise = model(denoised_pos, force) if use_forces else model(denoised_pos)
            denoised_pos = denoised_pos - predicted_noise  # Remove noise iteratively

        # Denormalize trajectories
        noisy_pos_np = application_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
        clean_pos_np = application_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
        denoised_pos_np = application_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()
        
        if postprocessing:
            # Apply smoothing using a moving average filter
            window_size = 20
            denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
            
            # Compute the offset using the average of the first 5 points difference
            offset = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
            denoised_pos_np += offset[:, np.newaxis, :]

        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
        overall_mean_diff = np.mean(np.abs(clean_pos_np - denoised_pos_np))

        mean_diffs_x.append(mean_diff_x)
        mean_diffs_y.append(mean_diff_y)
        mean_diffs_z.append(mean_diff_z)
        overall_mean_diffs.append(overall_mean_diff)

        # Save plot for each sequence
        fig, ax_traj = plt.subplots(1, 1, figsize=(12, 6))
        ax_traj.plot(clean_pos_np[0, :, 1], label='Clean (ground truth)', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 1], linestyle='--', label='Denoised (diffusion model)', linewidth=3.5, color='darkgreen')
        
        ax_traj.set_xlabel('Time Step', fontsize=16, fontweight='bold')
        ax_traj.set_ylabel(r'$\tilde{y}_o$ Position', fontsize=16, fontweight='bold')
        ax_traj.set_title(f'Clean vs Denoised pos in Y-direction - Seq {seq_idx+1}', fontsize=18, fontweight='bold')
        ax_traj.legend(fontsize=14)
        ax_traj.grid(True, linestyle="--", linewidth=1, alpha=0.7)
        ax_traj.tick_params(axis='both', labelsize=14, width=2.5, length=8)
        
        plot_filename = os.path.join(save_path, f"application_pos_seq_{seq_idx+1}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Print Mean Absolute Differences
    print(f"\nMean Absolute Differences Across {num_sequences} Sequences:")
    print(f"X-axis: {np.mean(mean_diffs_x):.6f}")
    print(f"Y-axis: {np.mean(mean_diffs_y):.6f}")
    print(f"Z-axis: {np.mean(mean_diffs_z):.6f}")
    print(f"Overall: {np.mean(overall_mean_diffs):.6f}")

    # Save results to file
    results_file = os.path.join(save_path, "application_results.txt")
    results_text = (
        f"\nMean Absolute Differences Across {num_sequences} Sequences:\n"
        f"X-axis: {np.mean(mean_diffs_x):.6f}\n"
        f"Y-axis: {np.mean(mean_diffs_y):.6f}\n"
        f"Z-axis: {np.mean(mean_diffs_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs):.6f}\n"
    )
    with open(results_file, "w") as file:
        file.write(results_text)
    print(f"Application inference results saved to {results_file}")
