import toml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from logzero import logger

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Model
from data import load_data, CovidChestxrayDataset

def check_grad(parameters):
    grad = 0
    cnt = 0
    for p in parameters:
        grad += p.grad.norm()
        cnt += 1
    return grad / cnt

def train():
    with open("config.toml") as f:
        config = toml.load(f)

    base_dir = config["data"]["base_dir"]
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    lr = config["train"]["lr"]
    betas = config["train"]["betas"]
    in_filters = config["model"]["in_filters"]
    image_size = config["model"]["image_size"]
    filters = config["model"]["filters"]
    num_classes = config["model"]["num_classes"]
    kernel_size = config["model"]["kernel_size"]
    padding = config["model"]["padding"]
    num_resblocks = config["model"]["num_resblocks"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_data(base_dir)
    train_records, test_records = train_test_split(records, test_size=0.2)

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(10, translate=[0.1, 0.1], shear=0.1),
        transforms.ColorJitter(brightness=0.7, contrast=0.7),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    trainset = CovidChestxrayDataset(train_records, base_dir, train_transform)
    testset = CovidChestxrayDataset(test_records, base_dir, test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    net = Model(in_filters, image_size, filters, kernel_size, padding, num_resblocks, num_classes)
    net.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=1e-2)

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        train_targets = []
        train_probs = []
        train_preds = []
        grad = 0
        for batch in trainloader:
            img, label = batch
            train_targets += label.numpy().tolist()
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = net(img)
            loss = criterion(pred, label)
            loss.backward()
            grad += check_grad(net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()
            train_preds += pred.cpu().detach().numpy().argmax(axis=1).tolist()
            train_probs += pred.cpu().detach().numpy()[:, 1].tolist()
        acc = accuracy_score(train_targets, train_preds)
        f1 = f1_score(train_targets, train_preds, average="macro")
        auc = roc_auc_score(train_targets, train_probs)
        logger.info(f"Epoch {epoch+1} Train loss {train_loss/len(trainloader):.5}, Acc {acc*100:.3}%, F1 {f1*100:.3}%, AUC {auc*100:.4}%, grad {grad/len(trainloader)}")
        net.eval()
        test_loss = 0
        test_targets = []
        test_preds = []
        test_probs = []
        for batch in testloader:
            img, label = batch
            test_targets += label.numpy().tolist()
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                pred = net(img)
                loss = criterion(pred, label)
            test_loss += loss.item()
            test_preds += pred.cpu().detach().numpy().argmax(axis=1).tolist()
            test_probs += pred.cpu().detach().numpy()[:, 1].tolist()

        acc = accuracy_score(test_targets, test_preds)
        f1 = f1_score(test_targets, test_preds, average="macro")
        auc = roc_auc_score(test_targets, test_probs)
        logger.info(f"Epoch {epoch+1} Test loss {test_loss/len(testloader):.5}, Acc {acc*100:.3}%, F1 {f1*100:.3}%, AUC {auc*100:.4}%")
    torch.save(net.state_dict, "net.pt")

if __name__ == "__main__":
    train()