import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import preprocess_data

# ===============================================================================
#                      PyTorch Lightning Training Functions 
# ===============================================================================
    
def train_lstm_baseline(X_train, y_train, X_val, y_val, X_test, y_test, exp_name):
    """
    Initializes, trains, and saves an LSTM baseline model.

    This function encapsulates the entire training pipeline for the LSTM model,
    including data module setup, model instantiation with best-found hyperparameters,
    and the configuration of callbacks for early stopping, checkpointing, and logging.

    Args:
        X_train, y_train: Training data and targets (as PyTorch tensors).
        X_val, y_val: Validation data and targets.
        X_test, y_test: Test data and targets.
        exp_name (str): A unique name for the experiment. Used for naming log files
                        and model checkpoints to keep results organized.
    """
    # Set a seed for reproducibility
    pl.seed_everything(42)

    # Determine the number of input and output features from the data shapes
    NUM_FEATURES = X_train.shape[2]
    OUT_FEATURES = 1 if y_train.ndim == 1 else y_train.shape[1]
    
    # Instantiate the DataModule
    data_module = TimeSeriesDataModule(X_train, y_train, X_val, y_val, X_test, y_test)

    # Instantiate the LightningModule with the best hyperparameters found via Optuna
    model = LSTMBaselineSystem(
        input_features=NUM_FEATURES,
        hidden_size=128,
        output_features=OUT_FEATURES,
        learning_rate=0.0011412674740790517,
        weight_decay=6.164039105250974e-06,
        dropout=0.1758228113490677
    )

    # --- Configure Callbacks ---
    # Stop training early if the validation loss stops improving
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    
    # Save the best version of the model based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'baseline-lstm-best-model-{{epoch:02d}}-{{val_loss:.4f}}-{exp_name}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Log all metrics to a CSV file for later analysis and plotting
    csv_logger = CSVLogger(save_dir="lightning_logs", name="lstm_logs", version=exp_name)

    # --- Configure the Trainer ---
    trainer = pl.Trainer(
        max_epochs=50,
        logger=csv_logger,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator='auto' # Automatically uses GPU if available
    )

    # --- Start Training ---
    trainer.fit(model, datamodule=data_module)

def train_transformer_system(X_train, y_train, X_val, y_val, X_test, y_test,
                             INPUT_LENGTH=168, OUTPUT_LENGTH=6, exp_name='', model_name='TransformerSystem'):
    """
    Initializes, trains, and saves a specified Transformer model.

    This function acts as a flexible trainer for the different Transformer architectures
    developed in this project. It selects the model based on `model_name` and uses
    the best-found hyperparameters for that architecture.

    Args:
        X_train, y_train: Training data and targets (as PyTorch tensors).
        X_val, y_val: Validation data and targets.
        X_test, y_test: Test data and targets.
        INPUT_LENGTH (int): The length of the input sequence (lookback window).
        OUTPUT_LENGTH (int): The length of the output sequence (forecast horizon).
        exp_name (str): A unique name for the experiment for logging and checkpoints.
        model_name (str): The name of the Transformer class to train. Must be one of
                          ['TransformerSystem', 'TransformerSystem2', 'TransformerSystem3'].
    """
    # Set a seed for reproducibility
    pl.seed_everything(42)
    
    # Determine the number of input features from the data shape
    NUM_FEATURES = X_train.shape[2] 
    
    # Instantiate the DataModule
    data_module = TimeSeriesDataModule(X_train, y_train, X_val, y_val, X_test, y_test)

    # --- Select and Instantiate the Model ---
    # This block selects the correct Transformer class based on the model_name
    # and initializes it with the best hyperparameters found via Optuna.
    if model_name == 'TransformerSystem':
        model = TransformerSystem(
            input_len=INPUT_LENGTH,
            patch_len=12,
            num_features=NUM_FEATURES,
            d_model=64,
            nhead=8,
            num_encoder_layers=5,
            dim_feedforward=512,
            dropout=0.24731555867746352,
            output_len=OUTPUT_LENGTH,
            learning_rate=0.00016370747430335437
        )
    elif model_name == 'TransformerSystem2':
        # Note: Hyperparameters for experimental models would be tuned separately
        model = TransformerSystem2(
            input_len=INPUT_LENGTH,
            patch_len=12,
            num_features=NUM_FEATURES,
            d_model=64,
            nhead=8,
            num_encoder_layers=5,
            dim_feedforward=512,
            dropout=0.24731555867746352,
            output_len=OUTPUT_LENGTH,
            learning_rate=0.00016370747430335437
        )
    elif model_name == 'TransformerSystem3':
        # Note: Hyperparameters for experimental models would be tuned separately
        model = TransformerSystem3(
            input_len=INPUT_LENGTH,
            patch_len=12,
            num_features=NUM_FEATURES,
            d_model=64,
            nhead=8,
            num_encoder_layers=5,
            dim_feedforward=512,
            dropout=0.24731555867746352,
            output_len=OUTPUT_LENGTH,
            learning_rate=0.00016370747430335437
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # --- Configure Callbacks ---
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    
    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'transformer-best-model-{{epoch:02d}}-{{val_loss:.4f}}-{exp_name}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    csv_logger = CSVLogger(save_dir="lightning_logs", name="transformer_logs", version=exp_name)

    # --- Configure the Trainer ---
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        logger=csv_logger,
        accelerator='auto'
    )

    # --- Start Training ---
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # Example usage of the training functions
    X_train, y_train, X_val, y_val, X_test, y_test, train_df = preprocess_data(predict_target='combined', output_length=6, input_length=48)
    train_transformer_system(X_train, y_train, X_val, y_val, X_test, y_test, exp_name='combined_lookback_48_forecast_6_model_1', OUTPUT_LENGTH=6, INPUT_LENGTH=48, model_name='TransformerSystem')
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_df = preprocess_data(predict_target='wind_onshore', output_length=6, input_length=168)
    train_lstm_baseline(X_train, y_train, X_val, y_val, X_test, y_test, exp_name='combined_lookback_168_forecast_6')