from omegaconf import DictConfig
import hydra

import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet_F
from utils.data_loading import BasicDataset, BasicDataset_val
from utils.dice_score import dice_loss
import torchvision.ops as ops
# import torchvision.ops.sigmoid_focal_loss as FocalLoss


torch.autograd.set_detect_anomaly = True

@hydra.main(config_name="config")
def train_model(cfg: DictConfig):
    # 0. Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet_F(cfg)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 )

    if cfg.model.load:
        state_dict = torch.load(cfg.model.load_dir, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model.load_dir}')

    model.to(device=device)

    # 1. Create dataset
    train_set = BasicDataset(cfg)
    val_set = BasicDataset_val(cfg)
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * cfg.data.val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=cfg.trainer.batch_size, num_workers=cfg.trainer.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    
    val_loader_args = dict(batch_size=cfg.predict.batch_size, num_workers=cfg.trainer.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='mycode', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=cfg.trainer.epochs, batch_size=cfg.trainer.batch_size, learning_rate=cfg.trainer.learning_rate,
             val_percent=cfg.data.val_percent,  img_scale=cfg.data.scale_input, amp=cfg.trainer.amp)
    )

    logging.info(f'''Starting training:  
        Epochs:          {cfg.trainer.epochs}
        Batch size:      {cfg.trainer.batch_size}
        Learning rate:   {cfg.trainer.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {cfg.data.scale_input}
        Mixed Precision: {cfg.trainer.amp}
    ''')


    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=cfg.trainer.learning_rate,
                              weight_decay=cfg.trainer.weight_decay,
                              momentum=cfg.trainer.momentum,
                              foreach=True
                             )
    amp = cfg.trainer.amp
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)    # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)


    global_step = 0

    # 5. Begin training
    epochs = cfg.trainer.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total = n_train, desc = f'Epoch {epoch} / {epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks  = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    mask_pred, img_resizer = model(images)

                    if model.n_classes == 1:
                        if cfg.trainer.FocalLoss:
                            loss = ops.sigmoid_focal_loss(mask_pred.squeeze(1), true_masks.float(), reduction='mean')
                        else:
                            criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
                            loss = criterion(mask_pred.squeeze(1), true_masks.float())

                        loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(mask_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0,3,1,2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // ( cfg.trainer.batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, niou = evaluate(model, val_loader, device, cfg.trainer.amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation niou: {}'.format(niou))

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                 'validation Dice': val_score,
                                'validation niou': niou,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        # Save model
        Path(cfg.model.save_dir).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict['mask_values'] = train_set.mask_values
        torch.save(state_dict, str(Path(cfg.model.save_dir) / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    train_model()








