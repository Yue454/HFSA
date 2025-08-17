import torch
from dataset import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from teacher import wide_resnet50_2
from student import de_wide_resnet50_2
from test import evaluation
from torch.nn import functional as F

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(teacher_maps, student_maps):
    """Calculates the cosine similarity loss between teacher and student feature maps."""
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(teacher_maps)):
        t_map_flat = teacher_maps[item].view(teacher_maps[item].shape[0], -1)
        s_map_flat = student_maps[item].view(student_maps[item].shape[0], -1)
        loss += torch.mean(1 - cos_loss(t_map_flat, s_map_flat))
    return loss

def train(_class_):
    """Main training loop for a given MVTec AD class."""
    print(f"Training for class: {_class_}")
    epochs =200
    learning_rate = 0.001
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = f'./mvtec/{_class_}/train'
    test_path = f'./mvtec/{_class_}'
    ckp_path = f'./checkpoints/wres50_{_class_}.pth'
    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    teacher_encoder, csab = wide_resnet50_2(pretrained=True)
    teacher_encoder = teacher_encoder.to(device).eval() # Teacher is always in eval mode
    csab = csab.to(device)

    student_decoder = de_wide_resnet50_2(pretrained=False).to(device)

    optimizer = torch.optim.Adam(list(student_decoder.parameters()) + list(csab.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    for epoch in range(epochs):
        csab.train()
        student_decoder.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)

            with torch.no_grad():
                teacher_features = teacher_encoder(img)
            compact_features = csab(teacher_features)
            student_features = student_decoder(compact_features)

            loss = loss_function(teacher_features, student_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(loss_list):.4f}')

        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(teacher_encoder, csab, student_decoder, test_dataloader, device)
            print(f'Validation -> Pixel AUROC: {auroc_px:.3f}, Sample AUROC: {auroc_sp:.3f}, AUPRO: {aupro_px:.3f}')
            torch.save({'csab': csab.state_dict(), 'decoder': student_decoder.state_dict()}, ckp_path)

    auroc_px, auroc_sp, aupro_px = evaluation(teacher_encoder, csab, student_decoder, test_dataloader, device)
    return auroc_px, auroc_sp, aupro_px

if __name__ == '__main__':


    item_list = ['carpet','grid','leather',  'tile', 'wood','pill', 'cable', 'capsule', 'screw', 'zipper', 'transistor',  'bottle', 'hazelnut',  'metal_nut', 'toothbrush']
    for i in item_list:
        train(i)