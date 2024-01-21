import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml
import random
import datetime
from tqdm import tqdm
from collections import namedtuple


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F


# A example for training SAM,
# user can specify the model name and the data root
# the class will load the model and train it with the given data


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.join = os.path.join
        self.data_root = self.join(os.path.dirname(__file__), data_root)
        self.gt_path = self.join(self.data_root, "gts")
        self.img_path = self.join(self.data_root, "imgs")

        print(f"gt path: {self.gt_path}")
        self.gt_path_files = sorted(
            glob.glob(self.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(self.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            self.join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


class SammTraining:
    def __init__(self, args, model_name: str) -> None:
        self.args = args
        self.model_name = model_name
        self.join = os.path.join

        pass

    def pre_processing(self):
        pass

    def load_pretrained_model(self):
        pass

    def train(self):
        # define cuda device and torch basics
        torch.manual_seed(2023)
        torch.cuda.empty_cache()

        if self.args.use_wandb:
            import wandb

            wandb.login()
            wandb.init(
                project=self.args.task_name,
                config={
                    "lr": self.args.lr,
                    "batch_size": self.args.batch_size,
                    "data_path": self.args.tr_npy_path,
                    "model_type": self.args.model_type,
                },
            )

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = self.join(
            self.args.work_dir, self.args.task_name + "-" + run_id
        )
        device = torch.device(self.args.device)

        self.load_pretrained_model()
        self.pre_processing()

        # Data loader
        train_ds = NpyDataset(data_root="data/npy/CT_Abd")
        print("Number of training samples: ", len(train_ds))
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )

        # Model
        if self.model_name == "MedSAM":
            sam_model = sam_model_registry[self.args.model_type](
                checkpoint=self.args.checkpoint
            )
            medsam_model = MedSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            ).to(device)

            medsam_model.train()

            print(
                "Number of total parameters: ",
                sum(p.numel() for p in medsam_model.parameters()),
            )  # 93735472

            print(
                "Number of trainable parameters: ",
                sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
            )  # 93729252

            img_mask_encoder_params = list(
                medsam_model.image_encoder.parameters()
            ) + list(medsam_model.mask_decoder.parameters())

            optimizer = torch.optim.AdamW(
                img_mask_encoder_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

            print(
                "Number of image encoder and mask decoder parameters: ",
                sum(p.numel() for p in img_mask_encoder_params if p.requires_grad),
            )

            seg_loss = monai.losses.DiceLoss(
                include_background=False, softmax=True, to_onehot_y=False
            )

            # cross entropy loss
            ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            # %% train
            num_epochs = self.args.num_epochs
            iter_num = 0
            losses = []
            best_loss = 1e10
            train_dataset = NpyDataset(self.args.tr_npy_path)
            train_dataset = NpyDataset(self.args.tr_npy_path)

            print("Number of training samples: ", len(train_dataset))
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

            start_epoch = 0
            if self.args.resume is not None:
                if os.path.isfile(self.args.resume):
                    ## Map model to be loaded to specified single GPU
                    checkpoint = torch.load(self.args.resume, map_location=device)
                    start_epoch = checkpoint["epoch"] + 1
                    medsam_model.load_state_dict(checkpoint["model"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(start_epoch, num_epochs):
                epoch_loss = 0
                for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
                    optimizer.zero_grad()
                    boxes_np = boxes.detach().cpu().numpy()
                    image, gt2D = image.to(device), gt2D.to(device)
                    if self.args.use_amp:
                        ## AMP
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            medsam_pred = medsam_model(image, boxes_np)
                            loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                                medsam_pred, gt2D.float()
                            )
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        medsam_pred = medsam_model(image, boxes_np)
                        loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                            medsam_pred, gt2D.float()
                        )
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    iter_num += 1

                epoch_loss /= step
                losses.append(epoch_loss)
                if self.args.use_wandb:
                    wandb.log({"epoch_loss": epoch_loss})
                print(
                    f'Time: {datetime.datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
                )
                ## save the latest model
                checkpoint = {
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    checkpoint, self.join(model_save_path, "medsam_model_latest.pth")
                )
                ## save the best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint, self.join(model_save_path, "medsam_model_best.pth")
                    )

                # plot loss
                plt.plot(losses, marker="*", color="purple", linestyle="-")
                plt.title("Dice + Cross Entropy Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim(0, epoch)
                plt.ylim(0, 1.1)
                plt.savefig(
                    self.join(model_save_path, self.args.task_name + "train_loss.png")
                )
                plt.close()

    def save_model(self):
        pass


def main():
    join = os.path.join

    # load config file
    config_path = join(os.path.dirname(__file__), "config/training.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_struct = namedtuple("Struct", config.keys())(*config.values())

    # define training object
    training_obj = SammTraining(config_struct, model_name="MedSAM")
    training_obj.train()


if __name__ == "__main__":
    main()
