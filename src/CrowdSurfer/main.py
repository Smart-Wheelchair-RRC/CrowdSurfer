from configuration import (
    Configuration,
    Mode,
    check_configuration,
    initialize_configuration,
)
from inference import InferencePipeline, Visualization
from training import PixelCNNTrainer, ScoringNetworkTrainer, VQVAETrainer


@initialize_configuration
def main(configuration: Configuration) -> None:
    check_configuration(configuration)

    if configuration.mode in {Mode.TRAIN_VQVAE, Mode.INFERENCE_VQVAE}:
        trainer = VQVAETrainer(configuration)
        if configuration.vqvae.checkpoint_path is not None:
            trainer.load_checkpoint(configuration.vqvae.checkpoint_path)
    elif configuration.mode in {Mode.TRAIN_PIXELCNN, Mode.INFERENCE_PIXELCNN}:
        trainer = PixelCNNTrainer(configuration)
        if configuration.pixelcnn.checkpoint_path is not None:
            trainer.load_checkpoint(configuration.pixelcnn.checkpoint_path)
    elif configuration.mode is Mode.TRAIN_SCORING_NETWORK:
        trainer = ScoringNetworkTrainer(configuration)
        if configuration.scoring_network.checkpoint_path is not None:
            trainer.load_checkpoint(configuration.scoring_network.checkpoint_path)

    if configuration.mode in {
        Mode.TRAIN_VQVAE,
        Mode.TRAIN_PIXELCNN,
        Mode.TRAIN_SCORING_NETWORK,
    }:
        trainer.train(
            num_epochs=configuration.trainer.num_epochs,
            epochs_per_save=configuration.trainer.epochs_per_save,
            logging_function=None,
        )
    elif configuration.mode in {Mode.INFERENCE_VQVAE, Mode.INFERENCE_PIXELCNN}:
        assert type(trainer) is VQVAETrainer or type(trainer) is PixelCNNTrainer
        for index in range(0, len(trainer.dataset), 150):
            trainer.inference(
                dataset_index=index,
                num_samples=50,
                save_plot_image=True,
                show_plot=False,
            )
    elif configuration.mode is Mode.INFERENCE_COMPLETE:
        pipeline = InferencePipeline(configuration)
        pipeline.run_all(plot=True, save_arrays=False)
    elif configuration.mode is Mode.VISUALIZE:
        visualization = Visualization(configuration)
        visualization.run_all()


if __name__ == "__main__":
    main()
