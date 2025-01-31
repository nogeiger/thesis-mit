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


#function to generate synthetic data
def generate_synthetic_trajectory_data(num_samples=14400, time_interval=10, seed=42):
    """
    Generate synthetic time-series data for robot trajectories, including position (x, y, z) 
    and forces (fx, fy, fz). The data is structured as required by IBM Granite-TTM.

    Args:
        num_samples (int): Number of time steps to generate.
        time_interval (int): Time interval in seconds between consecutive samples.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing timestamps, position (x, y, z), and force (fx, fy, fz).
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * time_interval) for i in range(num_samples)]

    # Generate smooth sinusoidal movements for trajectory (x, y, z)
    t = np.linspace(0, 50, num_samples)
    x = np.sin(t)#+ 0.05 * np.random.randn(num_samples)
    y = np.cos(t)#+ 0.05 * np.random.randn(num_samples)
    z = np.sin(t / 2) + 0.05 * np.random.randn(num_samples)

    # Generate synthetic force signals (correlated with movement)
    fx = np.gradient(x) + 0.01 * np.random.randn(num_samples)
    fy = np.gradient(y) + 0.01 * np.random.randn(num_samples)
    fz = np.gradient(z) + 0.01 * np.random.randn(num_samples)

    # Create and return DataFrame
    data = pd.DataFrame({
        "date": timestamps,
        "x": x,
        "y": y,
        "z": z,
        "fx": fx,
        "fy": fy,
        "fz": fz,
    })
    
    return data




#definition of fewshot training and evaluation
def fewshot_finetune_eval(
    TTM_MODEL_PATH,
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
    

    out_dir = os.path.join(save_dir, dataset_name)

    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

    # Data prep: Get dataset
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_val, dset_test = get_datasets(
        tsp, data, split_config, fewshot_fraction=fewshot_percent / 100, fewshot_location="first"
    )

    #load pretrained model
    finetune_forecast_model = get_model(
            TTM_MODEL_PATH,
            context_length=context_length,
            prediction_length=forecast_length,
            loss=loss,
            quantile=quantile,
    )

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable
    if learning_rate is None:
        learning_rate, finetune_forecast_model = optimal_lr_finder(
            finetune_forecast_model,
            dset_train,
            batch_size=batch_size,
        )
        print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    print(f"Using learning rate = {learning_rate}")
    #Define training arguments
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        seed=SEED,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),
    )
    #define trainer and train model
    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
    finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

    # Fine tune
    finetune_forecast_trainer.train()

    # Evaluation
    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)

    finetune_forecast_trainer.model.loss = "mse"  # fixing metric to mse for evaluation

    fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    print(fewshot_output)
    print("+" * 60)

    # get predictions

    predictions_dict = finetune_forecast_trainer.predict(dset_test)

    predictions_np = predictions_dict.predictions[0]

    print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    backbone_embedding = predictions_dict.predictions[1]

    print(backbone_embedding.shape)

    # plot
    plot_predictions(
        model=finetune_forecast_trainer.model,
        dset=dset_test,
        plot_dir=os.path.join(save_dir,dataset_name),
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
    PREDICTION_LENGTH = 96 # Granite-TTM-R2 supports forecast length upto 720 and Granite-TTM-R1 supports forecast length upto 96

    # Results/output directory
    OUT_DIR = "ttm_finetuned_models/"

    #generate synthetic data
    data = generate_synthetic_trajectory_data()

    # Dataset definitions
    TARGET_DATASET = "robot-data"
    #dtaset_path = "synthetic_trajectory_data.csv"
    timestamp_column = "date"
    id_columns = []  # mention the ids that uniquely identify a time-series.
    target_columns = ["x","y","z"] #outputs to predict
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
        "control_columns": ["fx","fy","fz"], #force as additional inputs
    }


    fewshot_finetune_eval(
        TTM_MODEL_PATH=TTM_MODEL_PATH,
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
        num_epochs=50,
        loss="mse",
        quantile=0.5,
    ) 


if __name__ == "__main__":
    main()


