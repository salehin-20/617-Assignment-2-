import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from dataset import GarbageDataset
from model import MultiModalNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = GarbageDataset(args.data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = MultiModalNet(num_classes=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, text_ids, labels in loader:
            images, text_ids = images.to(device), text_ids.to(device)
            logits = model(images, text_ids)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    classes = ["Black", "Blue", "Green", "Other"]
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))


if __name__ == "__main__":
    main()
