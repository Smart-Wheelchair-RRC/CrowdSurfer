from typing import Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .configuration import (
    Configuration,
    DatasetConfiguration,
    LiveConfiguration,
    Mode,
    PixelCNNConfiguration,
    ProjectionConfiguration,
    ScoringNetworkConfiguration,
    TrainerConfiguration,
    VQVAEConfiguration,
)


def initialize_configuration(
    function: Callable, configuration_name: str = "configuration"
) -> Callable:
    cs = ConfigStore.instance()
    cs.store(name="base_configuration", node=Configuration)
    cs.store(group="vqvae", name="base_vqvae", node=VQVAEConfiguration)
    cs.store(group="pixelcnn", name="base_pixelcnn", node=PixelCNNConfiguration)
    cs.store(
        group="scoring_network",
        name="base_scoring_network",
        node=ScoringNetworkConfiguration,
    )
    cs.store(group="trainer", name="base_trainer", node=TrainerConfiguration)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfiguration)
    cs.store(group="projection", name="base_projection", node=ProjectionConfiguration)
    cs.store(group="live", name="base_live", node=LiveConfiguration)

    return hydra.main(config_path="configuration", config_name=configuration_name)(
        function
    )


def check_configuration(configuration: Configuration) -> None:
    print(OmegaConf.to_yaml(configuration))
    if configuration.mode in {Mode.INFERENCE_VQVAE, Mode.TRAIN_PIXELCNN}:
        assert configuration.vqvae.checkpoint_path is not None
    elif configuration.mode in {Mode.INFERENCE_PIXELCNN, Mode.TRAIN_SCORING_NETWORK}:
        assert configuration.vqvae.checkpoint_path is not None
        assert configuration.pixelcnn.checkpoint_path is not None
    elif configuration.mode in {Mode.INFERENCE_COMPLETE, Mode.LIVE}:
        assert configuration.vqvae.checkpoint_path is not None
        assert configuration.pixelcnn.checkpoint_path is not None
        # assert configuration.scoring_network.checkpoint_path is not None
    elif configuration.mode in {Mode.VISUALIZE}:
        assert len(configuration.dataset.coefficient_configuration) == 1
