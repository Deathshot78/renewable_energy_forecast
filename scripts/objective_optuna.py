import optuna
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback
from preprocess import preprocess_data
from models import LSTMBaselineSystem, TimeSeriesDataset, TimeSeriesDataModule, PositionalEncoding, TransformerSystem, TransformerSystem2, TransformerSystem3

class ManualPruningCallback(Callback):
    """
    A PyTorch Lightning Callback that connects to an Optuna trial to enable pruning.
    
    This callback is a robust alternative to the official PyTorchLightningPruningCallback,
    avoiding potential version compatibility issues. It monitors a specified metric
    (e.g., 'val_loss') at the end of each validation epoch.
    """
    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        """
        Args:
            trial (optuna.trial.Trial): The Optuna trial object for the current run.
            monitor (str): The name of the metric to monitor for pruning (e.g., 'val_loss').
        """
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.step_count = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Hook called at the end of the validation loop.
        """
        # Get the latest value of the monitored metric
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None: 
            return
        
        # 1. Report the score to the Optuna trial
        self.trial.report(current_score.item(), self.step_count)
        self.step_count += 1
        
        # 2. Ask the trial if it should be pruned
        if self.trial.should_prune():
            # If Optuna's pruner decides this trial is unpromising, raise an exception
            # to stop the PyTorch Lightning training run.
            raise optuna.exceptions.TrialPruned(f"Trial pruned at epoch {self.step_count}.")
        
def lstm_objective(trial: optuna.trial.Trial, input_length: int, output_length: int, predict_target: str ) -> float:
    """
    The objective function for the Optuna hyperparameter search for the LSTM model.
    
    Optuna calls this function for each trial, suggesting a new set of hyperparameters.
    The function then trains a model with these parameters and returns the final
    validation loss, which Optuna aims to minimize.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object, used to suggest hyperparameters.
        input_length (int): The length of the input sequence.
        output_length (int): The length of the output sequence.
        predict_target (str): The target variable to predict ('combined', 'solar', 'wind', 'wind_onshore', 'wind_offshore').
    Returns:
        float: The best validation loss achieved during the trial.
    """
    # --- 1. Suggest Hyperparameters ---
    # Define the search space for Optuna to explore for each hyperparameter.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    # --- 2. Instantiate Model and Trainer ---
    # Create the model with the hyperparameters suggested for this specific trial
    
    # The DataModule remains the same for all trials
    X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(predict_target=predict_target, output_length=output_length, input_length=input_length)
    data_module = TimeSeriesDataModule(X_train, y_train, X_val, y_val, X_test, y_test)
    NUM_FEATURES = X_train.shape[2]
    if y_train.ndim == 1:
        OUT_FEATURES = 1
    else:
        OUT_FEATURES = y_train.shape[1]
    
    model = LSTMBaselineSystem(
        input_features=NUM_FEATURES,
        hidden_size=hidden_size,
        output_features=OUT_FEATURES,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout
    )
    
    # Configure a "silent" trainer for the study to keep the logs clean
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        logger=False, # Disable default loggers
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ManualPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor='val_loss', patience=5, verbose=False)
        ]
    )

    # --- 3. Run Training and Return Metric ---
    try:
        # Start the training run
        trainer.fit(model, datamodule=data_module)
    except optuna.exceptions.TrialPruned:
        # If our callback prunes the trial, re-raise the exception so Optuna knows
        raise
        
    # Get the final validation loss to return to Optuna
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss is not None else float('inf')

def transformer_objective(trial: optuna.trial.Trial, model_name: str, input_length: int, output_length: int, predict_target: str) -> float:
    """
    The objective function for the Optuna hyperparameter search for the Transformer models.

    This function allows tuning any of the Transformer architectures by specifying the
    `model_name`. It suggests hyperparameters, trains a model, and returns the validation
    loss for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        model_name (str): The name of the Transformer class to train 
                          ('TransformerSystem', 'TransformerSystem2', 'TransformerSystem3').
        input_length (int): The length of the input sequence.
        output_length (int): The length of the output sequence.
        predict_target (str): The target variable to predict ('combined', 'solar', 'wind', 'wind_onshore', 'wind_offshore').
    Returns:
        float: The best validation loss achieved during the trial.
    """
    # --- 1. Suggest Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 5)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
    patch_len = trial.suggest_categorical("patch_len", [8, 12, 16])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    # Add constraints to prune invalid trials early
    if d_model % nhead != 0 or input_length % patch_len != 0:
        raise optuna.exceptions.TrialPruned()
        
    # --- 2. Instantiate Model and DataModule ---
    # A dictionary mapping model names to their classes for cleaner code
    model_map = {
        'TransformerSystem': TransformerSystem,
        'TransformerSystem2': TransformerSystem2,
        'TransformerSystem3': TransformerSystem3
    }
    ModelClass = model_map.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model name: {model_name}")

    X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(predict_target=predict_target, output_length=output_length, input_length=input_length)
    data_module = TimeSeriesDataModule(X_train, y_train, X_val, y_val, X_test, y_test)
    
    NUM_FEATURES = X_train.shape[2] 
    
    model = ModelClass(
        input_len=input_length, patch_len=patch_len, num_features=NUM_FEATURES,
        d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward, output_len=output_length,
        dropout=dropout, learning_rate=learning_rate
    )
    
    

    # --- 3. Configure a "silent" Trainer ---
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ManualPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor='val_loss', patience=3, verbose=False)
        ]
    )

    # --- 4. Run Training and Return Metric ---
    try:
        trainer.fit(model, datamodule=data_module)
    except optuna.exceptions.TrialPruned:
        raise

    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss is not None else float('inf')

if __name__ == "__main__":
    # Example usage of the LSTM objective function
    input_length = 168  # Example input length
    output_length = 6   # Example output length
    predict_target = 'combined'  # Example target variable
    model_name = 'TransformerSystem'  # Example model name
    # Create an Optuna study and optimize the LSTM objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: transformer_objective(trial, model_name, input_length, output_length, predict_target), n_trials=20)

    print("Best trial:", study.best_trial)

    