import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
from data import build_dataset
import torchvision
from torch import nn
from networks import SimCLRNet
from utils import AvgMetricMeter
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier():
    def __init__(self, feature_type="backbone_features", k=20, metric="cosine", temp=0.07):
        self.metric = metric
        self.k = k
        self.temp = temp
        self.feature_type = feature_type

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        test_points, test_targets = [], []
        for batch in test_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")

            test_points.append(feature)
            test_targets.append(target)

        test_points = torch.cat(test_points, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        memory_bank = memory_bank.cpu().numpy()
        target_bank = target_bank.cpu().numpy()
        test_points = test_points.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        if self.metric == "cosine":
            def temp_scaled_cossim(cos_distance):
                # Cosine distance is defined as 1.0 minus the cosine similarity.
                # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
                # cos_distance = 1 - cos_sim -> cos_sim = -1 * cos_distance + 1
                cos_sim = -1.0 * cos_distance + 1.0
                return np.exp( cos_sim / self.temp)
            knn = KNeighborsClassifier(n_neighbors=self.k, metric="cosine", n_jobs=-1, algorithm='brute', weights=temp_scaled_cossim)
            knn.fit(memory_bank, target_bank)
            acc = knn.score(test_points, test_targets)
            return acc
        elif self.metric == "euclidean":
            knn = KNeighborsClassifier(n_neighbors=self.k, metric="euclidean", weights="uniform", n_jobs=-1)
            knn.fit(memory_bank, target_bank)
            acc = knn.score(test_points, test_targets)
            return acc
        else:
            raise ValueError("Unknown metric", self.metric)


class LogRegClassifier():
    def __init__(self, feature_type="backbone_features", solver="saga"):
        self.solver = solver
        self.feature_type = feature_type

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        test_points, test_targets = [], []
        for batch in test_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")

            test_points.append(feature)
            test_targets.append(target)

        test_points = torch.cat(test_points, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        memory_bank = memory_bank.cpu().numpy()
        target_bank = target_bank.cpu().numpy()
        test_points = test_points.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        logreg = LogisticRegression(solver=self.solver, n_jobs=-1, max_iter=1000)
        logreg.fit(memory_bank, target_bank)
        acc = logreg.score(test_points, test_targets)
        return acc


def main():

    parser = argparse.ArgumentParser(description='sklearn classifiers')

    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--temp', default=0.07, type=float)
    parser.add_argument('--metric', default="cosine", type=str,choices=["cosine", "euclidean"])
    parser.add_argument('--classifier', default="knn", type=str,choices=["knn", "logreg"])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--dataset', default='imagenet', type=str, choices=["imagenet", "imagenet100", "imagenette", "cifar10","cifar100"])
    parser.add_argument('--feature_type', default='backbone_features', type=str, choices=["backbone_features", "output_features"])
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--model_checkpoint_path', required=False)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--data_path', default="./data", type=str)

    args = parser.parse_args()

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(args)

    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split,
        random_state=args.random_state,
        data_path=args.data_path
    )

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Download weights from https://github.com/vturrisi/solo-learn
    # The  use https://solo-learn.readthedocs.io/en/latest/tutorials/offline_linear_eval.html 
    #  
    # create the backbone network
    # the first convolutional and maxpooling layers of the ResNet backbone
    # are adjusted to handle lower resolution images (32x32 instead of 224x224).
    backbone = torchvision.models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    #import pdb; pdb.set_trace()
    # load pretrained feature extractor
    state = torch.load("/home/erland/github/solo-learn/trained_models/cifar100/ressl/ressl-cifar100-mycgj38q-ep=999.ckpt")["state_dict"]
    #state = torch.load("/home/erland/iclr/saclr-main/sololearn/simsiamcifar100.ckpt")
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    torch.backends.cudnn.benchmark = True
    model = backbone.cuda()
    model.eval()

    if args.classifier == "knn":
        knn_classifier = KNNClassifier(
            feature_type=args.feature_type,
            k=args.k,
            metric=args.metric,
        )
        acc = knn_classifier.evaluate(model, trainloader, testloader)
        print("{}NN {} Accuracy: {:.2f}%".format(args.k, args.metric, acc * 100.0))

    elif args.classifier == "logreg":
        logreg_classifier = LogRegClassifier()
        acc = logreg_classifier.evaluate(model, trainloader, testloader)

        print("LogReg Accuracy: {:.2f}%".format(acc * 100.0))
    else:
        raise ValueError("Unknown classifier type")


if __name__ == "__main__":
    main()
