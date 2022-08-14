import logging
import os
from functools import partial

import hydra
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from config import ModelConfig
from model import build_model, train_loop_fn
from operations import (
    BasicPipeline,
    ModelDataset,
    PreTrainedEncoder,
    collate_batch,
    fos_get_data,
    get_pipelines,
)
from utils import get_system_device, set_up_folders

logging._level = logging.DEBUG


cs = ConfigStore.instance()
cs.store(name="finetuning_config", node=ModelConfig)


@hydra.main(config_path="../conf", config_name="application.yaml")
def main(cfg: ModelConfig):
    logging.info(cfg)
    set_up_folders([cfg.conf_data.dir, cfg.conf_model.dir])
    device = get_system_device(cfg.conf_model.gpu)
    data_path = os.path.join(cfg.conf_data.dir, cfg.conf_data.file[0])

    records, labels_encoder = fos_get_data(data_path)
    data_iter = ModelDataset(data=records)

    if cfg.model.build_own_encoder:
        pipeline = BasicPipeline(
            config=cfg,
            data_iter=data_iter,
            labels_encoder=labels_encoder,
        )
    else:
        pipeline = PreTrainedEncoder(
            config=cfg,
            data_iter=data_iter,
            labels_encoder=labels_encoder,
        )

    text_pipeline, label_pipeline = get_pipelines(pipeline)

    collate_batch_func = partial(
        collate_batch,
        text_pipeline=text_pipeline,
        label_pipeline=None,  # label_pipeline,
        device=device,
    )
    dataloader = DataLoader(
        data_iter,
        batch_size=cfg.conf_model.batch_size,
        shuffle=True,
        collate_fn=collate_batch_func,
    )

    model, optimizer = build_model(cfg, len(labels_encoder), device=device)
    train_loop_fn(
        model=model,
        data=dataloader,
        optimizer=optimizer,
        epochs=cfg.conf_model.epochs,
        data_size=len(records),
    )
    # Un here we can still quantized the model , see quantize_model repo
    # also we need to save the tokenizer as well # TODO

    torch.save(model.state_dict(), os.path.join(cfg.model.dir, f"{cfg.model.name}.pt"))
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(
        quantized_model.state_dict(),
        os.path.join(cfg.model.dir, f"{cfg.model.quantized_name}.pt"),
    )
    pipeline.save_tokenizer(os.path.join(cfg.model.dir, cfg.conf_tokenizer.folder_name))
    # then we can push it to MLFlow , with description , also during
    # training we can store experiments
    tokenizer = PreTrainedEncoder.load_tokenizer(
        os.path.join(cfg.model.dir, cfg.conf_tokenizer.folder_name)
    )


if __name__ == "__main__":
    main()
