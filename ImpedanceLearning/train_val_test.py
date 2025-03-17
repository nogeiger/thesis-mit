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
from utils import loss_function, quaternion_loss, add_noise, quaternion_inverse, quaternion_multiply, smooth_quaternions_slerp, quaternion_to_axis
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from datetime import datetime
from scipy.ndimage import uniform_filter1d



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
    lr_scheduler_patience = min(5, int(early_stop_patience * 0.32))  # Reduce LR after 1/3 of early stopping patience
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm to create a progress bar

        for batch_idx, (pos_0, pos, q_0, q, force, moment) in enumerate(tqdm(traindataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)):
            
            # Move data to device
            clean_pos = pos_0.to(device)
            complete_noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            complete_noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_q, noise_scale = add_noise(clean_pos, complete_noisy_pos, 
                                        clean_q, complete_noisy_q,
                                        force, moment,
                                        noiseadding_steps, beta_start, 
                                        beta_end, noise_with_force, add_gaussian_noise)
            
            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default noise

            #Calc actual noise for q: actual_noise_q = noisy_q * clean_q^-1
            # Compute actual noise in quaternion space
            actual_noise_q = quaternion_multiply(noisy_q, quaternion_inverse(clean_q))

            optimizer.zero_grad()

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_q, force, moment)
            
            else:
                predicted_noise = model(noisy_pos, noisy_q)


            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) +  10* quaternion_loss(predicted_noise[:,:,3:], actual_noise_q)
            #print(f"pos loss: {criterion(predicted_noise[:,:,0:3], actual_noise_pos) }")
            #print(f"q loss: {quaternion_loss(predicted_noise[:,:,3:], actual_noise_q)}")
            loss = loss / torch.clamp(noise_scale, min=1e-6) * 1000  # Normalize loss by noise scale
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
        for batch_idx, (pos_0, pos, q_0, q, force, moment)in enumerate(tqdm(dataloader, desc="Validating", leave=True)):
            clean_pos = pos_0.to(device)
            noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_q, noise_scale = add_noise(clean_pos, noisy_pos, clean_q, noisy_q,
                                        force, moment,
                                        max_noiseadding_steps, beta_start, beta_end, noise_with_force, add_gaussian_noise)

            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default: noise is the diff


            #Calc actual noise for q: actual_noise_q = noisy_q * clean_q^-1
            # Compute actual noise in quaternion space
            actual_noise_q = quaternion_multiply(noisy_q, quaternion_inverse(clean_q))

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_q, force, moment)
            else:
                predicted_noise = model(noisy_pos, noisy_q)

            # Calculate loss
            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) +  10 *quaternion_loss(predicted_noise[:,:,3:], actual_noise_q)
            loss = loss / torch.clamp(noise_scale, min=1e-6) * 1000
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
    mean_diffs_theta = []
    mean_diffs_axis_alpha = []


    for sample_idx, idx in enumerate(sample_indices):
        # Fetch a **random sample** instead of always using the first batch
        clean_pos, noisy_pos, clean_q, noisy_q, force, moment = val_data[idx]

        # Move data to the correct device
        clean_pos = clean_pos.unsqueeze(0).to(device)  # Add batch dimension
        noisy_pos = noisy_pos.unsqueeze(0).to(device)
        clean_q = clean_q.unsqueeze(0).to(device)
        noisy_q = noisy_q.unsqueeze(0).to(device)
        force = force.unsqueeze(0).to(device)
        moment = moment.unsqueeze(0).to(device)

        # Start iterative denoising
        denoised_pos = noisy_pos.clone()
        denoised_q = noisy_q.clone()

        for i in range(num_denoising_steps):
      
            predicted_noise = model(denoised_pos, denoised_q, force, moment) if use_forces else model(denoised_pos, denoised_q)
            denoised_pos = denoised_pos - predicted_noise[:,:,0:3]  # Remove noise iteratively
            denoised_q = quaternion_multiply(denoised_q, quaternion_inverse(predicted_noise[:,:,3:]))

    
        # Denormalize trajectories
        noisy_pos_np = val_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
        clean_pos_np = val_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
        denoised_pos_np = val_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()

        noisy_q_np = val_dataset.denormalize(noisy_q.detach().cpu(), "q").numpy()
        clean_q_np = val_dataset.denormalize(clean_q.detach().cpu(), "q_0").numpy()
        denoised_q_np = val_dataset.denormalize(denoised_q.detach().cpu(), "q_0").numpy()

        force_np = val_dataset.denormalize(force.detach().cpu(), "force").numpy()
        moment_np = val_dataset.denormalize(moment.detach().cpu(), "force").numpy()

        if postprocessing == True:
            #Preprocessing of denoise

            # Apply smoothing using a moving average filter
            window_size = 20  # Adjust the window size based on smoothing needs
            denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
            #smoothing for q using slerp with sliding window
            denoised_q_np = smooth_quaternions_slerp(torch.tensor(denoised_q_np), window_size=window_size, smoothing_factor=0.5).numpy()

            #remove offses
            # Compute the offset using the average of the first 5 points difference
            offset_pos = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
            # Apply the offset to all points in the denoised pos
            denoised_pos_np += offset_pos[:, np.newaxis, :]
            #for quaternions
            q_offset = quaternion_multiply(clean_q[:, 0, :], quaternion_inverse(denoised_q[:, 0, :]))
            # Apply offset correction to the entire sequence
            for t in range(denoised_q.shape[1]):  # Iterate over timesteps
                denoised_q[:, t, :] = quaternion_multiply(q_offset, denoised_q[:, t, :])

            denoised_q_np = denoised_q


        

        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
        overall_mean_diff_pos = np.mean(np.abs(clean_pos_np - denoised_pos_np))

        # Append mean differences to lists
        mean_diffs_pos_x.append(mean_diff_x)
        mean_diffs_pos_y.append(mean_diff_y)
        mean_diffs_pos_z.append(mean_diff_z)
        overall_mean_diffs_pos.append(overall_mean_diff_pos)

        # Compute angular differences theta of quaternions
        # Ensure quaternions are on CPU and converted to NumPy
        denoised_q_np = denoised_q.detach().cpu().numpy()  # Move to CPU and convert to NumPy
        clean_q_np = clean_q.detach().cpu().numpy()  # Move to CPU and convert to NumPy
        # Compute rotation angle theta from quaternions
        theta_clean = 2 * np.arccos(np.clip(clean_q_np[0,:, 0], -1.0, 1.0))  # First component is cos(theta/2)
        # Normalize the entire denoised quaternion
        denoised_q_np /= np.linalg.norm(denoised_q_np, axis=-1, keepdims=True)
        theta_denoised = 2 * np.arccos(np.clip(denoised_q_np[0,:, 0], -1.0, 1.0))
        # Ensure angles are in degrees first (optional if already in radians)
        theta_clean_deg = np.degrees(theta_clean)
        theta_denoised_deg = np.degrees(theta_denoised)
        # Compute angular difference with wrap-around handling
        theta_error = np.abs((theta_clean_deg - theta_denoised_deg + 180) % 360 - 180)
        # Compute mean error
        mean_theta_error = np.mean(theta_error)
        mean_diffs_theta.append(mean_theta_error)



        # Compute axis-angle representation and loss of alpha for quaternions
        # Extract rotation axes (u) from quaternions
        u_clean = quaternion_to_axis(clean_q_np[0,:, :])  # Shape (T, 3)
        u_denoised = quaternion_to_axis(denoised_q_np[0,:, :])  # Shape (T, 3)
        # Compute angular difference between axes (for calc of cosine - cosine(theta>1) values leads to instability)
        dot_product_u = np.einsum('ij,ij->i', u_clean, u_denoised)  # Batch-wise dot product
        # Fix: Take absolute value to handle u and -u equivalence
        dot_product_u = np.clip(np.abs(dot_product_u), -1.0, 1.0)  

        # Compute angle difference
        alpha_error = np.arccos(dot_product_u)  # Angle difference in radians

        # Convert to degrees
        alpha_error_deg = np.degrees(alpha_error)

        # Compute mean axis error
        mean_alpha_error = np.mean(alpha_error_deg)
        mean_diffs_axis_alpha.append(mean_alpha_error) 


        # Create a separate figure for each sample
        fig, ax_traj = plt.subplots(1, 1, figsize=(12, 6))  # Wider figure for better visibility

        # Plot clean vs denoised pos (Y-axis only) with thicker lines
        ax_traj.plot(clean_pos_np[0, :, 0], label='Clean (ground truth) x', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 0], linestyle='--', label='Denoised (diffusion model) x', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_pos_np[0, :, 1], label='Clean (ground truth) y', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 1], linestyle='--', label='Denoised (diffusion model) y', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_pos_np[0, :, 2], label='Clean (ground truth) z', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 2], linestyle='--', label='Denoised (diffusion model) z', linewidth=3.5, color='darkgreen')

   
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
        f"Theta: {np.mean(mean_diffs_theta):.6f}\n"
        f"Alpha: {np.mean(mean_diffs_axis_alpha):.6f}\n"
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
    mean_diffs_pos_x, mean_diffs_pos_y, mean_diffs_pos_z, overall_mean_diffs_pos = [], [], [], []
    mean_diffs_theta = []
    mean_diffs_axis_alpha = []

    # Persistent storage for all data
    all_data = []

    for seq_idx in range(num_sequences):
        # Fetch the sequence in order
        clean_pos, noisy_pos, clean_q, noisy_q, force, moment = application_data[seq_idx]

        # Move data to the correct device
        clean_pos = clean_pos.unsqueeze(0).to(device)  # Add batch dimension
        noisy_pos = noisy_pos.unsqueeze(0).to(device)
        clean_q = clean_q.unsqueeze(0).to(device)
        noisy_q = noisy_q.unsqueeze(0).to(device)
        force = force.unsqueeze(0).to(device)
        moment = moment.unsqueeze(0).to(device)


        # Start iterative denoising
        denoised_pos = noisy_pos.clone()
        denoised_q = noisy_q.clone()
        for _ in range(num_denoising_steps):
            predicted_noise = model(denoised_pos, denoised_q ,force, moment) if use_forces else model(denoised_pos, denoised_q)
            denoised_pos = denoised_pos - predicted_noise[:,:,0:3]  # Remove noise iteratively
            denoised_q = quaternion_multiply(denoised_q, quaternion_inverse(predicted_noise[:,:,3:]))



        # Denormalize trajectories
        noisy_pos_np = application_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
        clean_pos_np = application_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
        denoised_pos_np = application_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()

        noisy_q_np = application_dataset.denormalize(noisy_q.detach().cpu(), "q").numpy()
        clean_q_np = application_dataset.denormalize(clean_q.detach().cpu(), "q_0").numpy()
        denoised_q_np = application_dataset.denormalize(denoised_q.detach().cpu(), "q_0").numpy()

        force_np = application_dataset.denormalize(force.detach().cpu(), "force").numpy()
        moment_np = application_dataset.denormalize(moment.detach().cpu(), "force").numpy()
        
        if postprocessing == True:
            #Preprocessing of denoise

            # Apply smoothing using a moving average filter
            window_size = 20  # Adjust the window size based on smoothing needs
            denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
            #smoothing for q using slerp with sliding window
            denoised_q_np = smooth_quaternions_slerp(torch.tensor(denoised_q_np), window_size=window_size, smoothing_factor=0.5).numpy()

            #remove offses
            # Compute the offset using the average of the first 5 points difference
            offset_pos = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
            # Apply the offset to all points in the denoised pos
            denoised_pos_np += offset_pos[:, np.newaxis, :]
            #for quaternions
            q_offset = quaternion_multiply(clean_q[:, 0, :], quaternion_inverse(denoised_q[:, 0, :]))
            # Apply offset correction to the entire sequence
            for t in range(denoised_q.shape[1]):  # Iterate over timesteps
                denoised_q[:, t, :] = quaternion_multiply(q_offset, denoised_q[:, t, :])

            denoised_q_np = denoised_q


        # Store all data in a structured format for txt file for matlab/robot
        T = clean_pos_np.shape[1]  # Sequence length
        time_array = np.arange(T) * 0.005  # Ensure time increments correctly

        for t in range(T):
            all_data.append([
                int(seq_idx), float(time_array[t]),
                *map(float, clean_pos_np[0, t, :]),  # Expands (x, y, z)
                *map(float, denoised_pos_np[0, t, :]),
                *map(float, noisy_pos_np[0, t, :]),
                *map(float, clean_q_np[0, t, :]),  # Expands (w, x, y, z)
                *map(float, denoised_q_np[0, t, :]),
                *map(float, noisy_q_np[0, t, :]),
                *map(float, force_np[0, t, :]),  # Expands (fx, fy, fz)
                *map(float, moment_np[0, t, :])  # Expands (mx, my, mz)
            ])

        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
        overall_mean_diff_pos = np.mean(np.abs(clean_pos_np - denoised_pos_np))

        # Append mean differences to lists
        mean_diffs_pos_x.append(mean_diff_x)
        mean_diffs_pos_y.append(mean_diff_y)
        mean_diffs_pos_z.append(mean_diff_z)
        overall_mean_diffs_pos.append(overall_mean_diff_pos)

        # Compute angular differences theta of quaternions
        # Ensure quaternions are on CPU and converted to NumPy
        denoised_q_np = denoised_q.detach().cpu().numpy()  # Move to CPU and convert to NumPy
        clean_q_np = clean_q.detach().cpu().numpy()  # Move to CPU and convert to NumPy
        # Compute rotation angle theta from quaternions
        theta_clean = 2 * np.arccos(np.clip(clean_q_np[0,:, 0], -1.0, 1.0))  # First component is cos(theta/2)
        # Normalize the entire denoised quaternion
        denoised_q_np /= np.linalg.norm(denoised_q_np, axis=-1, keepdims=True)
        theta_denoised = 2 * np.arccos(np.clip(denoised_q_np[0,:, 0], -1.0, 1.0))
        # Ensure angles are in degrees first (optional if already in radians)
        theta_clean_deg = np.degrees(theta_clean)
        theta_denoised_deg = np.degrees(theta_denoised)
        # Compute angular difference with wrap-around handling
        theta_error = np.abs((theta_clean_deg - theta_denoised_deg + 180) % 360 - 180)
        # Compute mean error
        mean_theta_error = np.mean(theta_error)
        mean_diffs_theta.append(mean_theta_error)

        # Compute axis-angle representation and loss of alpha for quaternions
        # Extract rotation axes (u) from quaternions
        u_clean = quaternion_to_axis(clean_q_np[0,:, :])  # Shape (T, 3)
        u_denoised = quaternion_to_axis(denoised_q_np[0,:, :])  # Shape (T, 3)
        # Compute angular difference between axes (for calc of cosine - cosine(theta>1) values leads to instability)
        dot_product_u = np.einsum('ij,ij->i', u_clean, u_denoised)  # Batch-wise dot product
        # Fix: Take absolute value to handle u and -u equivalence
        dot_product_u = np.clip(np.abs(dot_product_u), -1.0, 1.0)  

        # Compute angle difference
        alpha_error = np.arccos(dot_product_u)  # Angle difference in radians

        # Convert to degrees
        alpha_error_deg = np.degrees(alpha_error)

        # Compute mean axis error
        mean_alpha_error = np.mean(alpha_error_deg)
        mean_diffs_axis_alpha.append(mean_alpha_error) 




        # Create a separate figure for each sample
        fig, ax_traj = plt.subplots(1, 1, figsize=(12, 6))  # Wider figure for better visibility

        # Plot clean vs denoised pos (Y-axis only) with thicker lines
        ax_traj.plot(clean_pos_np[0, :, 0], label='Clean (ground truth) x', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 0], linestyle='--', label='Denoised (diffusion model) x', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_pos_np[0, :, 1], label='Clean (ground truth) y', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 1], linestyle='--', label='Denoised (diffusion model) y', linewidth=3.5, color='darkgreen')
        ax_traj.plot(clean_pos_np[0, :, 2], label='Clean (ground truth) z', linewidth=3.5, color='darkblue')
        ax_traj.plot(denoised_pos_np[0, :, 2], linestyle='--', label='Denoised (diffusion model) z', linewidth=3.5, color='darkgreen')

   
        # Customize plot appearance with bold labels and increased font size
        ax_traj.set_xlabel('Time Step', fontsize=16, fontweight='bold')
        ax_traj.set_ylabel(r'$\tilde{y}_o$ Position', fontsize=16, fontweight='bold')  # Y-label with tilde notation
        ax_traj.set_title(f'Clean vs denoised zero force pos in y-direction - Sample {seq_idx+1}', 
                        fontsize=18, fontweight='bold')

        ax_traj.legend(fontsize=14)

        # Make grid lines more visible
        ax_traj.grid(True, linestyle="--", linewidth=1, alpha=0.7)

        # Increase tick label size and make ticks thicker
        ax_traj.tick_params(axis='both', labelsize=14, width=2.5, length=8)

        # Define save path for the plot
        plot_filename = os.path.join(save_path, f"pos_sample_{seq_idx+1}.png")

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
        f"\nMean Absolute Differences Across Samples:\n"
        f"X-axis: {np.mean(mean_diffs_pos_x):.6f}\n"
        f"Y-axis: {np.mean(mean_diffs_pos_y):.6f}\n"
        f"Z-axis: {np.mean(mean_diffs_pos_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs_pos):.6f}\n\n"
        f"Theta: {np.mean(mean_diffs_theta):.6f}\n"
        f"Alpha: {np.mean(mean_diffs_axis_alpha):.6f}\n"
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


        # Convert to DataFrame and save once at the end
    columns = [
        "Seq_Index", "Time",
        "Clean_X", "Clean_Y", "Clean_Z",
        "Denoised_X", "Denoised_Y", "Denoised_Z",
        "Noisy_X", "Noisy_Y", "Noisy_Z",
        "Clean_Q_W", "Clean_Q_X", "Clean_Q_Y", "Clean_Q_Z",
        "Denoised_Q_W", "Denoised_Q_X", "Denoised_Q_Y", "Denoised_Q_Z",
        "Noisy_Q_W", "Noisy_Q_X", "Noisy_Q_Y", "Noisy_Q_Z",
        "Force_X", "Force_Y", "Force_Z",
        "Moment_X", "Moment_Y", "Moment_Z"
    ]

    df = pd.DataFrame(all_data, columns=columns)
    output_file = os.path.join(save_path, "inference_results.txt")

    # Save in tab-separated format
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Results saved to {output_file}")
