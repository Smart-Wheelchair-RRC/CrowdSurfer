import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class TrainerMode(Enum):
    TRAIN = auto()
    INFERENCE = auto()


def _ceildiv(a, b):
    return -(a // -b)


@dataclass
class TrainingMetadata:
    batch_size: int
    learning_rate: float
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.batch_size = state_dict["batch_size"]
        self.learning_rate = state_dict["learning_rate"]
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.best_loss = state_dict["best_loss"]

    def increment_epoch(self):
        self.epoch += 1

    def increment_step(self):
        self.step += 1

    def reset_step(self):
        self.step = 0


class Trainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        results_directory: str,
        learning_rate: float,
        batch_size: int,
        mode: TrainerMode = TrainerMode.TRAIN,
        validation_dataset: Optional[Dataset] = None,
        use_safetensors_for_saving: bool = False,
        dataloader_num_workers: int = 12,
        dataloader_pin_memory: bool = True,
    ):
        self.mode = mode
        self.accelerator = Accelerator()
        self.use_safe_tensors = use_safetensors_for_saving

        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(parents=True, exist_ok=True)

        self.metadata = TrainingMetadata(batch_size=batch_size, learning_rate=learning_rate)

        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.model = model

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

        if self.validation_dataset is not None:
            self.validation_dataloader = DataLoader(
                self.validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=dataloader_num_workers,
                pin_memory=dataloader_pin_memory,
            )

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.accelerator.register_for_checkpointing(self.metadata)

        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.DEBUG)
        _stream_handler = logging.StreamHandler()
        _stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        _stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(_stream_handler)

        if self.mode == TrainerMode.TRAIN:
            _file_handler = logging.FileHandler(self.results_directory.joinpath("training.log"))
            _file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            )
            _file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(_file_handler)

        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

        self.skipped_dataloader = None

    def _default_logging_function(
        self,
        epoch: int,
        step: Optional[int],
        loss: float,
        validation_loss: Optional[float],
    ):
        self.logger.debug(f"Epoch: {epoch}, Step: {step}, Loss: {loss}, Validation Loss: {validation_loss}")

    def _save_checkpoint(self, name: str):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(
            str(self.results_directory.joinpath(name)),
            safe_serialization=self.use_safe_tensors,
        )
        with logging_redirect_tqdm(loggers=[self.logger]):
            self.logger.info(f"Saved checkpoint {name}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        dataset_length: Optional[int] = None,
    ):
        total_data_points = len(self.dataset) if dataset_length is None else dataset_length  # type: ignore
        current_batch_size = self.metadata.batch_size
        current_learning_rate = self.metadata.learning_rate

        # Load the checkpoint
        self.accelerator.load_state(checkpoint_path)

        total_traversed_datapoints = self.metadata.epoch * total_data_points + (
            (self.metadata.step * current_batch_size)
            if (total_data_points % current_batch_size == 0 or self.metadata.step == 0)
            else ((self.metadata.step - 1) * current_batch_size + total_data_points % current_batch_size)
        )

        # Calculate the true total epochs and steps traversed so far
        total_epochs_traversed = int(total_traversed_datapoints // total_data_points)
        total_current_steps_traversed = int(
            _ceildiv(total_traversed_datapoints % total_data_points, current_batch_size)
        )

        self.skipped_dataloader = (
            self.accelerator.skip_first_batches(self.dataloader, total_current_steps_traversed)
            if total_current_steps_traversed > 0
            else self.skipped_dataloader
        )

        self.metadata.epoch = total_epochs_traversed
        self.metadata.batch_size = current_batch_size
        self.metadata.learning_rate = current_learning_rate
        self.metadata.reset_step()

        self.logger.info(
            f"Checkpoint {checkpoint_path} loaded with best loss: {self.metadata.best_loss} and epoch: {self.metadata.epoch}"
        )

    @property
    def configuration(self):
        return {
            "batch_size": self.metadata.batch_size,
            "learning_rate": self.metadata.learning_rate,
        }

    @abstractmethod
    def forward(self, data: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        output: Union[Tensor, Tuple[Tensor, ...]],
        data: Dict[str, Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def train(
        self,
        num_epochs: int = 10,
        epochs_per_save: int = 10,
        logging_function: Optional[Callable[[int, Optional[int], float, Optional[float]], None]] = None,
    ):
        assert self.mode == TrainerMode.TRAIN, "Trainer is not in training mode"

        self.model.train()

        total_epochs = self.metadata.epoch + num_epochs

        self.logger.info(f"Training model with configuration: {self.configuration}")
        self.logger.info(f"Dataset length: {len(self.dataset)}")  # type: ignore

        # Define logging functions
        logging_functions = [self._default_logging_function]
        if logging_function is not None:
            logging_functions.append(logging_function)  # type: ignore

        # Define progress bars
        progress_bar = tqdm(
            initial=self.metadata.epoch,
            total=total_epochs,
            desc="Epoch",
            position=0,
            leave=True,
            unit="epoch",
        )
        inner_progress_bar = tqdm(
            initial=self.metadata.step,
            total=len(self.dataloader),
            desc="Batch",
            position=1,
            leave=True,
            unit="batch",
        )

        # Main training loop
        while self.metadata.epoch < total_epochs:
            # Choose dataloader
            dataloader = self.skipped_dataloader if self.skipped_dataloader is not None else self.dataloader
            inner_progress_bar.total = len(dataloader)

            # Inner training loop
            epoch_loss = 0
            for data in dataloader:  # type: ignore
                self.optimizer.zero_grad()

                output = self.forward(data)

                loss = self.loss(output, data)

                epoch_loss += loss.item()

                self.accelerator.backward(loss)

                self.optimizer.step()

                self.metadata.increment_step()

                inner_progress_bar.set_postfix(loss=f"{loss.item()}")
                inner_progress_bar.update(1)

                for logging_function in logging_functions:
                    logging_function(
                        self.metadata.epoch,
                        self.metadata.step,
                        loss.item(),
                        None,
                    )
            epoch_loss /= len(dataloader)

            # Reset skipped dataloader
            if self.skipped_dataloader is not None:
                self.skipped_dataloader = None

            # Validation
            validation_loss = None

            if self.validation_dataset is not None:
                validation_loss = 0

                with torch.no_grad():
                    for data in self.validation_dataloader:
                        data: Dict[str, Tensor]
                        data = {key: value.to(self.accelerator.device) for key, value in data.items()}
                        output = self.forward(data)
                        loss = self.loss(output, data)
                        validation_loss += loss.item()

                validation_loss /= len(self.validation_dataloader)

            comparison_loss = validation_loss if validation_loss is not None else epoch_loss
            is_best = comparison_loss < self.metadata.best_loss

            # Update progress bars and metadata
            self.metadata.reset_step()
            self.metadata.increment_epoch()

            inner_progress_bar.reset()
            progress_bar.update(1)

            progress_bar.set_postfix(
                **(
                    {
                        "loss": f"{epoch_loss}" + " (Best)" if is_best else "",
                    }
                    if validation_loss is None
                    else {
                        "loss": f"{epoch_loss}",
                        "val_loss": f"{validation_loss}" + " (Best)" if is_best else "",
                    }
                )  # type: ignore
            )

            # Checkpointing
            if is_best:
                self.metadata.best_loss = comparison_loss
                self._save_checkpoint("best_checkpoint")

            if self.metadata.epoch % epochs_per_save == 0:
                self._save_checkpoint(f"checkpoint_{self.metadata.epoch}")

            # Logging
            for logging_function in logging_functions:
                logging_function(
                    self.metadata.epoch,
                    None,
                    epoch_loss,
                    validation_loss,
                )

        inner_progress_bar.close()
        with logging_redirect_tqdm(loggers=[self.logger]):
            self.logger.info("Training Complete!")
