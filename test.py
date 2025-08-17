import torch
from dataset import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from teacher import resnet18, wide_resnet50_2
from student import de_resnet18, de_wide_resnet50_2
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, auc
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
import pickle


def cal_anomaly_map(teacher_features, student_features, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])

    a_map_list = []
    for i in range(len(teacher_features)):
        fs = student_features[i]
        ft = teacher_features[i]
        a_map = 1 - F.cosine_similarity(ft, fs)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation(teacher_encoder, csab, student_decoder, dataloader, device, _class_=None):
    teacher_encoder.eval()
    csab.eval()
    student_decoder.eval()

    gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, aupro_list = [], [], [], [], []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            teacher_features = teacher_encoder(img)
            compact_features = csab(teacher_features)
            student_features = student_decoder(compact_features)

            anomaly_map, _ = cal_anomaly_map(teacher_features, student_features, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int), anomaly_map[np.newaxis, :, :]))

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    mean_aupro = round(np.mean(aupro_list), 3) if aupro_list else 0.0

    return auroc_px, auroc_sp, mean_aupro


def test(_class_):
    """Runs a test on a saved model checkpoint."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device} for class: {_class_}")

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = f'../mvtec/{_class_}'
    ckp_path = f'./checkpoints/wres50_{_class_}.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    teacher_encoder, csab = wide_resnet50_2(pretrained=True)
    teacher_encoder.to(device).eval()
    csab.to(device)
    student_decoder = de_wide_resnet50_2(pretrained=False).to(device)

    ckp = torch.load(ckp_path)
    # Load state dicts using the new key 'csab'
    csab.load_state_dict(ckp['csab'])
    student_decoder.load_state_dict(ckp['decoder'])

    auroc_px, auroc_sp, aupro_px = evaluation(teacher_encoder, csab, student_decoder, test_dataloader, device, _class_)
    print(f'{_class_}: Pixel AUROC={auroc_px}, Sample AUROC={auroc_sp}, AUPRO={aupro_px}')
    return auroc_px


def visualization(_class_):
    """Generates and displays visualizations of anomaly detection results."""
    print(f"Visualizing for class: {_class_}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = f'../mvtec/{_class_}'
    ckp_path = f'./checkpoints/wres50_{_class_}.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    teacher_encoder, csab = wide_resnet50_2(pretrained=True)
    teacher_encoder.to(device).eval()
    csab.to(device).eval()
    student_decoder = de_wide_resnet50_2(pretrained=False).to(device).eval()

    ckp = torch.load(ckp_path)
    csab.load_state_dict(ckp['csab'])
    student_decoder.load_state_dict(ckp['decoder'])

    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue

            img = img.to(device)
            teacher_features = teacher_encoder(img)
            compact_features = csab(teacher_features)
            student_features = student_decoder(compact_features)

            anomaly_map, _ = cal_anomaly_map([teacher_features[-1]], [student_features[-1]], img.shape[-1],
                                             amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map_heatmap = cvt2heatmap(ano_map * 255)

            img_vis = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0], cv2.COLOR_RGB2BGR)
            img_vis = np.uint8(min_max_norm(img_vis) * 255)

            overlay = show_cam_on_image(img_vis, ano_map_heatmap)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axs[1].set_title('Anomaly Overlay')
            axs[1].axis('off')
            axs[2].imshow(gt.squeeze().cpu().numpy(), cmap='gray')
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')
            plt.show()




def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    if np.sum(masks) == 0:
        return 1.0

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th if num_th > 0 else 1e-6

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for bi_amap, mask in zip(binary_amaps, masks):
            labeled_mask = measure.label(mask)
            for region in measure.regionprops(labeled_mask):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = bi_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum() if inverse_masks.sum() > 0 else 0

        new_row = pd.DataFrame([{"pro": mean(pros) if pros else 0, "fpr": fpr, "threshold": th}])
        df = pd.concat([df, new_row], ignore_index=True)

    df = df[df["fpr"] < 0.3]
    if df.empty or df["fpr"].max() == 0: return 0.0

    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc