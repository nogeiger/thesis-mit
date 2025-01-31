import math
import os
import tempfile

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TimeSeriesPreprocessor, TrackingCallback, count_parameters, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

#generate synthetic data
def generate_noisy_trajectory_data(num_samples=14400, time_interval=10, noise_std=0.1, seed=42):
    """
    Generate synthetic trajectory data where (x, y, z) is corrupted by noise.
    The model learns to predict the noise instead of the clean trajectory.

    Args:
        num_samples (int): Number of time steps.
        time_interval (int): Time interval in seconds.
        noise_std (float): Standard deviation of Gaussian noise.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: A DataFrame containing timestamps, noisy (x, y, z), forces (fx, fy, fz), and noise.
    """
    np.random.seed(seed)

    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * time_interval) for i in range(num_samples)]

    t = np.linspace(0, 50, num_samples)
    clean_x = np.sin(t)
    clean_y = np.cos(t)
    clean_z = np.sin(t / 2)

    # Generate Gaussian noise
    noise_x = noise_std * np.random.randn(num_samples)
    noise_y = noise_std * np.random.randn(num_samples)
    noise_z = noise_std * np.random.randn(num_samples)

    # Noisy positions (Inputs)
    noisy_x = clean_x + noise_x
    noisy_y = clean_y + noise_y
    noisy_z = clean_z + noise_z

    # Generate force signals (remain unchanged)
    fx = np.gradient(clean_x) + 0.01 * np.random.randn(num_samples)
    fy = np.gradient(clean_y) + 0.01 * np.random.randn(num_samples)
    fz = np.gradient(clean_z) + 0.01 * np.random.randn(num_samples)

        # Create and return DataFrame
    data = pd.DataFrame({
        "date": timestamps,
        "clean_x": clean_x,
        "clean_y": clean_y,
        "clean_z": clean_z,
        "noisy_x": noisy_x,
        "noisy_y": noisy_y,
        "noisy_z": noisy_z,
        "noise_x": noise_x,
        "noise_y": noise_y,
        "noise_z": noise_z,
        "fx":fx,
        "fy":fy,
        "fz": fz,
    })   
    
    return data



def fewshot_finetune_eval(
    model_path,
    data,
    context_length,
    dataset_name,
    batch_size,
    save_dir,
    column_specifiers,
    split_config,
    learning_rate=None,
    forecast_length=96,
    fewshot_percent=5,
    freeze_backbone=True,
    num_epochs=50,
    loss="mse",
    quantile=0.5,
):
    """
    Fine-tune the IBM Granite-TTM model to predict noise instead of positions.

    Args:
        model_path (str): Path to the IBM Granite-TTM model.
        data (pd.DataFrame): Input dataset.
        context_length (int): Number of historical steps to consider.
        dataset_name (str): Name of the dataset.
        batch_size (int): Training batch size.
        save_dir (str): Directory for saving the model.
        column_specifiers (dict): Dataset structure.
        split_config (dict): Train-validation-test split.
        learning_rate (float or None): Learning rate.
        forecast_length (int): Number of future steps to predict.
        fewshot_percent (int): Percentage of data for fine-tuning.
        freeze_backbone (bool): If True, freezes model backbone.
        num_epochs (int): Number of epochs.
        loss (str): Loss function.
        quantile (float): Quantile value for quantile loss.

    Returns:
        None
    """
    out_dir = os.path.join(save_dir, dataset_name)

    print(f"🔍 Running few-shot fine-tuning ({fewshot_percent}%) for NOISE prediction")

    tsp = TimeSeriesPreprocessor(
        timestamp_column=column_specifiers["timestamp_column"],
        id_columns=column_specifiers["id_columns"],
        target_columns=column_specifiers["target_columns"], # We predict noise instead of position
        observable_columns=column_specifiers["observable_columns"],
       #control_columns=["noisy_x", "noisy_y", "noisy_z", "fx", "fy", "fz"],  # Inputs include noisy positions and forces
        #conditional_columns: List[str] = [],
        #static_categorical_columns: List[str] = [],
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_val, dset_test = get_datasets(
        tsp, data, split_config, fewshot_fraction=fewshot_percent / 100, fewshot_location="first"
    )

    model = get_model(model_path=model_path, 
                      context_length=context_length, 
                      prediction_length=forecast_length, 
                      num_input_channels=tsp.num_input_channels,
                      decoder_mode="mix_channel",
                      prediction_channel_indices=tsp.prediction_channel_indices,
    )


    if freeze_backbone:
        print("🔒 Freezing model backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    if learning_rate is None:
        learning_rate, model = optimal_lr_finder(model, dset_train, batch_size)
        print(f"⚡ Optimal Learning Rate: {learning_rate}")

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        optimizers=(AdamW(model.parameters(), lr=learning_rate), None),
    )

    trainer.train()

    print("✅ Model evaluation...")
    results = trainer.evaluate(dset_test)
    print(results)

    print("📊 Plotting predictions...")
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=os.path.join(save_dir, dataset_name),
        plot_prefix="test_fewshot",
        indices=[685, 118, 902, 1984, 894, 967, 304, 57, 265, 1015],
        channel=0,
    )


##############################
#main function 
def main():

    #Model selection
    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
    # TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"
    # TTM_MODEL_PATH = "ibm-research/ttm-research-r2"

    #model paramters
    CONTEXT_LENGTH = 512 # Context length, Or Length of the history.
    PREDICTION_LENGTH = CONTEXT_LENGTH #For diffusion process the noise prediction length is same as context length of trajectory and force lenght

    # Results/output directory
    OUT_DIR = "ttm_finetuned_models/"

    #generate synthetic data
    data = generate_noisy_trajectory_data()

    # Dataset definitions
    TARGET_DATASET = "robot-data"
    timestamp_column = "date"
    id_columns = []  # mention the ids that uniquely identify a time-series.
    target_columns = ["noise_x", "noise_y", "noise_z"] #output names
    observable_columns=["noisy_x", "noisy_y", "noisy_z","fx","fy","fz"]
    #c!!!!!!!!!!!!change this split by a factor!!!!!!!!!!
    split_config = {
        "train": [0, 8640],         #first samples for train
        "valid": [8640, 11520],     #next samples for validation
        "test": [                   #remaining samples for test
            11520,
            14400,
        ],
    }

    #specify the columns of the dataset
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "observable_columns": observable_columns,
    }


    fewshot_finetune_eval(
        model_path=TTM_MODEL_PATH,
        data=data,
        context_length=CONTEXT_LENGTH,
        dataset_name=TARGET_DATASET,
        batch_size=64,
        save_dir=OUT_DIR,
        column_specifiers=column_specifiers,
        split_config=split_config,
        learning_rate=0.001,
        forecast_length=PREDICTION_LENGTH,
        fewshot_percent=5,
        freeze_backbone=True,
        num_epochs=500,
        loss="mse",
        quantile=0.5,
    ) 



if __name__ == "__main__":
    main()


