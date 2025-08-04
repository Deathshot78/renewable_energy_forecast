import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import math

from preprocess import preprocess_data
from models import LSTMBaselineSystem, TimeSeriesDataset, TimeSeriesDataModule, PositionalEncoding, TransformerSystem, TransformerSystem2, TransformerSystem3
# ===============================================================================
#                   PyTorch Dataset and Lightning DataModule
#     These classes handle the loading and preparation of data for the model.
# ===============================================================================
class TimeSeriesDataset(Dataset):
    """
    A simple PyTorch Dataset for time series data.
    It takes the pre-sequenced input (X) and target (y) arrays and serves
    them up sample by sample.
    """
    def __init__(self, X, y):
        """
        Args:
            X (np.array or torch.Tensor): The input sequences. 
                                          Shape: (num_samples, input_length, num_features)
            y (np.array or torch.Tensor): The target sequences.
                                          Shape: (num_samples, output_length) or (num_samples,)
        """
        self.X = X
        self.y = y

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (input and target) from the dataset.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            tuple: A tuple containing the input tensor and the target tensor.
        """
        return self.X[idx], self.y[idx]

class TimeSeriesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule to encapsulate all data-related steps.
    This module handles the creation of Datasets and DataLoaders for the
    training, validation, and testing sets.
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64, num_workers=0):
        """
        Args:
            X_train, y_train: Training data and targets.
            X_val, y_val: Validation data and targets.
            X_test, y_test: Test data and targets.
            batch_size (int): The number of samples per batch.
            num_workers (int): The number of subprocesses to use for data loading.
        """
        super().__init__()
        # Store the raw data arrays
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Assigns the datasets for each stage (fit, validate, test).
        This is called on every GPU in a distributed setup.
        """
        self.train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        self.val_dataset = TimeSeriesDataset(self.X_val, self.y_val)
        self.test_dataset = TimeSeriesDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# ===============================================================================
#                       PyTorch Lightning Model Definitions 
# ===============================================================================

class LSTMBaselineSystem(pl.LightningModule):
    """
    A PyTorch Lightning module for the LSTM baseline model.
    This class encapsulates the model architecture, the forward pass logic,
    and the training, validation, and test steps.
    """
    def __init__(self, input_features, hidden_size, output_features, learning_rate=1e-4, weight_decay=1e-3, dropout=0.2):
        """
        Args:
            input_features (int): The number of features in the input data.
            hidden_size (int): The number of features in the LSTM hidden state.
            output_features (int): The number of output steps to predict (forecast horizon).
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 regularization) for the optimizer.
            dropout (float): The dropout rate for regularization.
        """
        super().__init__()
        # This saves all hyperparameters to self.hparams, making them accessible later
        self.save_hyperparameters()

        # --- Model Architecture ---
        # LSTM layer that processes the input sequence
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True, # Expects input tensors of shape (batch, seq, feature)
        )
        
        # Dropout layer for regularization after the LSTM
        self.dropout = nn.Dropout(dropout) 
        
        # Final fully-connected layer to map the LSTM output to the desired forecast horizon
        self.linear = nn.Linear(hidden_size, output_features)
        
        # The loss function used for training
        self.loss_fn = nn.L1Loss() # Mean Absolute Error

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor. Shape: (batch, seq_len, features)
            
        Returns:
            torch.Tensor: The model's prediction. Shape: (batch, output_len)
        """
        # Pass the input through the LSTM layer
        # We only care about the output, not the final hidden and cell states (_)
        lstm_out, _ = self.lstm(x)
        
        # We take the output of the very last time step from the LSTM sequence
        last_time_step_out = lstm_out[:, -1, :]
        
        # Apply dropout for regularization
        dropout_out = self.dropout(last_time_step_out)
        
        # Pass the result through the final linear layer to get the prediction
        return self.linear(dropout_out)

    def _get_extra_metrics(self, y_hat, y):
        """
        Helper function to calculate additional evaluation metrics (sMAPE and Accuracy).
        
        Args:
            y_hat (torch.Tensor): The model's predictions.
            y (torch.Tensor): The true target values.
            
        Returns:
            tuple: A tuple containing the sMAPE and accuracy values.
        """
        # sMAPE (Symmetric Mean Absolute Percentage Error) - A robust percentage error
        epsilon = 1e-8
        numerator = torch.abs(y_hat - y)
        denominator = (torch.abs(y) + torch.abs(y_hat)) / 2 + epsilon
        smape = torch.mean(numerator / denominator) * 100
        
        # Threshold Accuracy - Percentage of predictions within a 5% tolerance
        relative_tolerance = 0.05
        absolute_tolerance = 0.01
        absolute_error = torch.abs(y_hat - y)
        mask = torch.abs(y) > epsilon
        relative_error = torch.zeros_like(y)
        if torch.any(mask):
            relative_error[mask] = absolute_error[mask] / torch.abs(y[mask])
        
        accurate_predictions = (relative_error < relative_tolerance) | (absolute_error < absolute_tolerance)
        accuracy = torch.mean(accurate_predictions.float()) * 100

        return smape, accuracy

    def training_step(self, batch, batch_idx):
        """Defines a single training step."""
        x, y = batch
        y_hat_raw = self(x)
        
        # Squeeze the output if necessary to match the label's shape (for single-step forecasts)
        y_hat = y_hat_raw.squeeze() if y_hat_raw.ndim > y.ndim else y_hat_raw
        
        # Calculate the loss
        loss = self.loss_fn(y_hat, y)
        
        # Log the training loss for monitoring
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines a single validation step."""
        x, y = batch
        y_hat_raw = self(x)
        
        # Squeeze the output to match the label's shape
        y_hat = y_hat_raw.squeeze() if y_hat_raw.ndim > y.ndim else y_hat_raw
        
        # Calculate loss and all extra metrics
        loss = self.loss_fn(y_hat, y)
        smape, acc = self._get_extra_metrics(y_hat, y)
        
        # Log all validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_smape', smape, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Defines a single test step."""
        x, y = batch
        y_hat_raw = self(x)

        # Squeeze the output to match the label's shape
        y_hat = y_hat_raw.squeeze() if y_hat_raw.ndim > y.ndim else y_hat_raw
        
        # Calculate loss and all extra metrics for the final report
        loss = self.loss_fn(y_hat, y)
        smape, acc = self._get_extra_metrics(y_hat, y)
        self.log('test_mae', loss)
        self.log('test_smape', smape)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        # AdamW is a robust optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # CosineAnnealingLR smoothly decreases the learning rate over epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, # The number of epochs to complete one cosine cycle
            eta_min=1e-6                  # The minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }      

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    Since Transformers do not have a built-in sense of sequence order, this
    module adds a unique encoding to each position in the sequence, allowing
    the model to understand the order of the patches.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimensionality of the model's embeddings.
            dropout (float): The dropout rate.
            max_len (int): The maximum possible sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix using sine and cosine functions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer so it's part of the model's state but not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_len, d_model)
        """
        # Add the positional encoding, slicing it to the input's sequence length.
        # It will be broadcasted across the batch dimension.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSystem(pl.LightningModule):
    """
    Base Transformer model inspired by PatchTST.
    This version uses a "feature-mixing" approach where all features within a patch
    are flattened together before being fed into the model. The prediction head
    is a simple flattening of the encoder output.
    This was the best-performing model in the experiments.
    """
    def __init__(self, 
                 input_len: int, patch_len: int, num_features: int, d_model: int, 
                 nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 output_len: int, dropout: float = 0.1, learning_rate: float = 1e-4):
        super().__init__()
        if input_len % patch_len != 0:
            raise ValueError("input_len must be divisible by patch_len for this model version.")
        self.save_hyperparameters()
        
        # --- Model Architecture ---
        # 1. Patching and Embedding
        num_patches = input_len // patch_len
        self.input_projection = nn.Linear(patch_len * num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_patches)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Prediction Head (Simple Flattening)
        self.prediction_head = nn.Linear(num_patches * d_model, output_len)
        
        # 5. Loss Function
        self.loss_fn = nn.L1Loss() # Mean Absolute Error

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, input_len, num_features)
        
        Returns:
            torch.Tensor: Prediction tensor. Shape: (batch_size, output_len)
        """
        batch_size = x.shape[0]
        num_patches = self.hparams.input_len // self.hparams.patch_len
        
        # 1. Patching: Reshape into patches, mixing features together
        patches = x.reshape(batch_size, num_patches, self.hparams.patch_len * self.hparams.num_features)
        
        # 2. Embedding: Project patches to the model's dimension
        patches_embedded = self.input_projection(patches)
        
        # 3. Add Positional Encoding
        pos_encoded = self.pos_encoder(patches_embedded)
        
        # 4. Pass through Transformer Encoder
        encoder_out = self.transformer_encoder(pos_encoded)
        
        # 5. Prediction: Flatten the encoder output and pass through the final layer
        flattened_out = encoder_out.reshape(batch_size, -1)
        predictions = self.prediction_head(flattened_out)
        
        return predictions
        
    # The _get_extra_metrics, training_step, validation_step, and configure_optimizers
    # methods are identical to the LSTMBaselineSystem and are thus inherited conceptually.
    # For a standalone script, you would copy them here.
    def _get_extra_metrics(self, y_hat, y):
        epsilon = 1e-8
        numerator = torch.abs(y_hat - y)
        denominator = (torch.abs(y) + torch.abs(y_hat)) / 2 + epsilon
        smape = torch.mean(numerator / denominator) * 100
        absolute_error = torch.abs(y_hat - y)
        mask = torch.abs(y) > epsilon
        relative_error = torch.zeros_like(y)
        if torch.any(mask):
            relative_error[mask] = absolute_error[mask] / torch.abs(y[mask])
        accurate_predictions = (relative_error < 0.05) | (absolute_error < 0.01)
        accuracy = torch.mean(accurate_predictions.float()) * 100
        return smape, accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        smape, acc = self._get_extra_metrics(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_smape', smape, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}

class TransformerSystem2(pl.LightningModule):
    """
    Experimental Transformer model using a "Channel-Independent" patching strategy.
    Each feature (channel) is treated as its own time series, patched independently,
    and then all patch embeddings are combined into a long sequence for the encoder.
    """
    def __init__(self, input_len: int, patch_len: int, num_features: int, d_model: int, 
                 nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 output_len: int, dropout: float = 0.1, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.patch_len = patch_len
        num_patches = input_len // patch_len
        
        # --- Model Architecture ---
        # 1. Patching and Embedding (Channel-Independent)
        self.input_projection = nn.Linear(self.patch_len, d_model)
        
        # 2. Positional Encoding (for a longer sequence)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=(num_patches * num_features))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Prediction Head
        self.prediction_head = nn.Linear((num_patches * num_features) * d_model, output_len)
        
        # 5. Loss Function
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        """
        Defines the forward pass for the channel-independent model.
        """
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        patches_embedded = self.input_projection(patches)
        transformer_input = patches_embedded.reshape(batch_size, -1, self.hparams.d_model)
        pos_encoded = self.pos_encoder(transformer_input)
        encoder_out = self.transformer_encoder(pos_encoded)
        flattened_out = encoder_out.reshape(batch_size, -1)
        predictions = self.prediction_head(flattened_out)
        return predictions

    def _get_extra_metrics(self, y_hat, y):
        epsilon = 1e-8
        numerator = torch.abs(y_hat - y)
        denominator = (torch.abs(y) + torch.abs(y_hat)) / 2 + epsilon
        smape = torch.mean(numerator / denominator) * 100
        absolute_error = torch.abs(y_hat - y)
        mask = torch.abs(y) > epsilon
        relative_error = torch.zeros_like(y)
        if torch.any(mask):
            relative_error[mask] = absolute_error[mask] / torch.abs(y[mask])
        accurate_predictions = (relative_error < 0.05) | (absolute_error < 0.01)
        accuracy = torch.mean(accurate_predictions.float()) * 100
        return smape, accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        smape, acc = self._get_extra_metrics(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_smape', smape, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
    
class TransformerSystem3(pl.LightningModule):
    """
    Experimental Transformer model using the base "feature-mixing" patching but with
    a more sophisticated, two-stage prediction head.
    """
    def __init__(self, 
                 input_len: int, patch_len: int, num_features: int, d_model: int, 
                 nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 output_len: int, dropout: float = 0.1, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # --- Model Architecture ---
        # 1. Patching and Embedding (Base Version)
        num_patches = input_len // patch_len
        self.input_projection = nn.Linear(patch_len * num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_patches)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Prediction Head (Sophisticated Two-Stage)
        self.head_linear1 = nn.Linear(d_model, 1)
        self.head_flatten = nn.Flatten(start_dim=1)
        self.head_linear2 = nn.Linear(num_patches, output_len)
        
        # 5. Loss Function
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        """
        Defines the forward pass for the model with the sophisticated head.
        """
        batch_size = x.shape[0]
        
        # 1. Patching and Embedding (Base Version)
        num_patches = self.hparams.input_len // self.hparams.patch_len
        patches = x.reshape(batch_size, num_patches, self.hparams.patch_len * self.hparams.num_features)
        patches_embedded = self.input_projection(patches)
        
        # 2. Encode
        pos_encoded = self.pos_encoder(patches_embedded)
        encoder_out = self.transformer_encoder(pos_encoded)
        
        # 3. Predict using the two-stage head
        x = self.head_linear1(encoder_out)
        x = self.head_flatten(x)
        predictions = self.head_linear2(x)
        
        return predictions

    def _get_extra_metrics(self, y_hat, y):
        epsilon = 1e-8
        numerator = torch.abs(y_hat - y)
        denominator = (torch.abs(y) + torch.abs(y_hat)) / 2 + epsilon
        smape = torch.mean(numerator / denominator) * 100
        absolute_error = torch.abs(y_hat - y)
        mask = torch.abs(y) > epsilon
        relative_error = torch.zeros_like(y)
        if torch.any(mask):
            relative_error[mask] = absolute_error[mask] / torch.abs(y[mask])
        accurate_predictions = (relative_error < 0.05) | (absolute_error < 0.01)
        accuracy = torch.mean(accurate_predictions.float()) * 100
        return smape, accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_raw = self(x)
        if y_hat_raw.ndim > y.ndim:
            y_hat = y_hat_raw.squeeze(-1)
        else:
            y_hat = y_hat_raw
        loss = self.loss_fn(y_hat, y)
        smape, acc = self._get_extra_metrics(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_smape', smape, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
    
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