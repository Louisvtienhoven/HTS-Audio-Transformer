import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import librosa

from utils import create_folder, dump_config
import anomaly_detection_config as config
from sed_model import SEDWrapper
from data_generator import ESC_Dataset
from model.htsat_in_chans_2 import HTSAT_Swin_Transformer

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Set up paths
    workspace = "./anomaly_detection_workspace"
    dataset_path = os.path.join(workspace, "Data")
    checkpoint_path = os.path.join(workspace, "ckpt")
    esc_raw_path = os.path.join(dataset_path, 'raw')
    resample_path = os.path.join(dataset_path, 'resample_12s')
    savedata_path = os.path.join(dataset_path, 'data.npy')

    # Create required directories
    for p in [workspace, dataset_path, checkpoint_path, esc_raw_path, resample_path]:
        create_path(p)
    print("[INFO] Directories created or verified.")

    # Paths to audio and metadata
    meta_path = os.path.join(esc_raw_path, 'Master', 'meta', 'meta_final_dataset_train_ver.csv')
    audio_path = os.path.join(esc_raw_path, 'Master', 'Dataset_final_12_second')
    print(f"[CHECK] meta_path = {meta_path}")
    print(f"[CHECK] audio_path = {audio_path}")

    # Check paths
    if not os.path.exists(meta_path):
        print(f"[ERROR] Metadata file not found: {meta_path}")
    else:
        print("[INFO] Metadata file found.")

    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio folder not found: {audio_path}")
    else:
        print("[INFO] Audio folder found.")

    if not (os.path.exists(meta_path) and os.path.exists(audio_path)):
        print("[FATAL] One or more required paths are missing. Exiting.")
        exit(1)

    # Resample audio
    meta = np.loadtxt(meta_path, delimiter=',', dtype='str', skiprows=1)
    audio_list = os.listdir(audio_path)
    print("[INFO] Resampling audio files...")
    for f in audio_list:
        full_f = os.pa(audio_path, f)
        resample_f = os.path.join(resample_path, f)
        if not os.path.exists(resample_f):
            os.system(f'sox -V1 {full_f} -r 32000 {resample_f}')
    print("[INFO] Resampling complete.")

    # Build dataset
    print("[INFO] Building dataset...")
    output_dict = [[] for _ in range(5)]
    for label in meta:
        name, fold, target = label[0], int(label[1]), int(label[2])
        y, sr = librosa.load(os.path.join(resample_path, name), sr=None)
        output_dict[fold - 1].append({"name": name, "target": target, "waveform": y})
    np.save(savedata_path, output_dict)
    print("[INFO] Dataset saved to data.npy")

    # Lightning DataModule
    class DataPrep(pl.LightningDataModule):
        def __init__(self, train_dataset, eval_dataset, device_num):
            super().__init__()
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.device_num = device_num

        def train_dataloader(self):
            sampler = DistributedSampler(self.train_dataset, shuffle=False) if self.device_num > 1 else None
            return DataLoader(self.train_dataset, batch_size=config.batch_size // self.device_num,
                              shuffle=(sampler is None), sampler=sampler, num_workers=config.num_workers)

        def val_dataloader(self):
            sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
            return DataLoader(self.eval_dataset, batch_size=config.batch_size // self.device_num,
                              shuffle=False, sampler=sampler, num_workers=config.num_workers)

    # Setup
    device_num = torch.cuda.device_count()
    print("[INFO] Number of GPUs detected:", device_num)
    print("[INFO] Each batch size:", config.batch_size // device_num)
    print("[INFO] Learning rate:", config.learning_rate)
    print("[INFO] Hop size:", config.hop_size)
    print("[INFO] Patch size:", config.htsat_patch_size)
    print("[INFO] Window size:", config.htsat_window_size)
    print("[INFO] spec size:", config.htsat_spec_size)

    full_dataset = np.load(savedata_path, allow_pickle=True)
    dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=False)
    eval_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=True)
    data_module = DataPrep(dataset, eval_dataset, device_num)
    print("[INFO] Data module initialized.")

    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)
        print("[INFO] Experiment folders and config dumped.")

    # Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="acc", filename='l-{epoch:d}-{acc:.3f}', save_top_k=5, mode="max"
    )

    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir=checkpoint_dir,
        gpus=device_num,
        max_epochs=config.max_epoch,
        auto_lr_find=True,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback],
        accelerator="ddp" if device_num > 1 else None,
        num_sanity_val_steps=0,
        resume_from_checkpoint=None,
        replace_sampler_ddp=False,
        gradient_clip_val=1.0
    )
    print("[INFO] Trainer initialized.")

    # Model
    sed_model = HTSAT_Swin_Transformer(
        pretrained=False,
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=config.htsat_in_chans,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )

    model = SEDWrapper(sed_model=sed_model, config=config, dataset=dataset)
    print("[INFO] Model wrapper created.")

    if config.resume_checkpoint is not None:
        print("[INFO] Loading checkpoint from", config.resume_checkpoint)
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        for key in ["sed_model.head.weight", "sed_model.head.bias",
                    "sed_model.tscam_conv.weight", "sed_model.tscam_conv.bias"]:
            ckpt["state_dict"].pop(key, None)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    print("[INFO] Starting training...")
    trainer.fit(model, data_module)
    print("[INFO] Training completed.")
