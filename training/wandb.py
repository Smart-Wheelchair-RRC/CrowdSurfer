from typing import Optional

import wandb


# Setup wandb
def setup_wandb(
    project_name: str,
    experiment_name: str,
    wandb_api_key: str,
    entity: str,
):
    wandb.login(key=wandb_api_key)
    wandb.init(
        project=project_name,
        name=experiment_name,
        entity=entity,
    )


def wandb_logging_function(
    epoch: int,
    step: Optional[int],
    loss: float,
    validation_loss: Optional[float],
):
    wandb.log(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "validation_loss": validation_loss,
        }
    )
