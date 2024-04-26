import hydra
from omegaconf import DictConfig

from trainer import Trainer
from loguru import logger
import sys
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
