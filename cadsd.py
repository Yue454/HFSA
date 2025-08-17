import torch
from dataset import get_data_transforms, CAD_SD_Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from teacher import wide_resnet50_2
from student import de_wide_resnet50_2
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from test import cal_anomaly_map


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_function(teacher_maps, student_maps):

    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(teacher_maps)):
        t_map_flat = teacher_maps[item].view(teacher_maps[item].shape[0], -1)
        s_map_flat = student_maps[item].view(student_maps[item].shape[0], -1)
        loss += torch.mean(1 - cos_loss(t_map_flat, s_map_flat))
    return loss


def loco_evaluation(teacher_encoder, csab, student_decoder, dataloader, device):

    teacher_encoder.eval()
    csab.eval()
    student_decoder.eval()

    gt_list_sp = []
    pr_list_sp = []

    with torch.no_grad():
        for img, label, _, _ in dataloader:
            img = img.to(device)

            teacher_features = teacher_encoder(img)
            compact_features = csab(teacher_features)
            student_features = student_decoder(compact_features)

            anomaly_map, _ = cal_anomaly_map(teacher_features, student_features, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.append(label.item())
            pr_list_sp.append(np.max(anomaly_map))

    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    return auroc_sp


def train(_class_):
    print(f"Training for CAD-SD class: {_class_}")
    epochs = 200
    learning_rate = 0.001
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data_transform, _ = get_data_transforms(image_size, image_size)

    train_path = f'./{_class_}/train'
    test_path = f'./{_class_}'
    ckp_path = f'./checkpoints/wres50_cadsd_{_class_}.pth'
    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = CAD_SD_Dataset(root=test_path, transform=data_transform, phase="test")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    teacher_encoder, csab = wide_resnet50_2(pretrained=True)
    teacher_encoder.to(device).eval()
    csab.to(device)
    student_decoder = de_wide_resnet50_2(pretrained=False).to(device)

    optimizer = torch.optim.Adam(list(student_decoder.parameters()) + list(csab.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

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
            auroc_sp = loco_evaluation(teacher_encoder, csab, student_decoder, test_dataloader, device)
            print(f'Sample AUROC: {auroc_sp:.3f}')
            torch.save({'csab': csab.state_dict(), 'decoder': student_decoder.state_dict()}, ckp_path)

    auroc_sp = loco_evaluation(teacher_encoder, csab, student_decoder, test_dataloader, device)
    return auroc_sp


if __name__ == '__main__':
    item_list = ['csd']
    for i in item_list:
        train(i)