# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class PseudoLabelCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, feature_dim, epsilon=0.1, lamda=0.003, similarity='cosine'):
        super().__init__()
        self._n_classes = n_classes
        self._feature_dim = feature_dim
        self._epsilon = epsilon
        self._lamda = lamda
        self._similarity = similarity
        print('| Using Pseudo Label Cross Entropy Loss, epsilon is {}, similarity is {}'.format(self._epsilon,
                                                                                                self._similarity))

        self.centers = nn.Parameter(torch.randn(n_classes, feature_dim))  # (200, 512)
        self.center_func = centerFunc.apply

    def init_centers(self, feature_distribution):
        """
        mean_feature = get_init_center(self._net, self._train_loader, self._n_classes, n_dim_feature=512)
        criterion.init_centers(mean_feature.cuda())
        """
        self.centers.data = feature_distribution
        assert self.centers.shape[1] == self._feature_dim
        assert self.centers.is_cuda

    def get_pseudo_label(self):
        center = self.centers.detach()
        if self._similarity == 'euclidean':
            # Euclidean distance based similarity
            c_norm = (center ** 2).sum(dim=1).view(-1, 1)  # (200, 1)
            dist = c_norm + c_norm.t() - 2.0 * torch.mm(center, center.t())
            dist = torch.sqrt(torch.clamp(dist, 0.0, float('inf')))
            euc_similarity = 1 / (dist + 1)
            return F.softmax(euc_similarity, dim=1)
        elif self._similarity == 'cosine':
            # Cosine Similarity
            dot_prod_matrix = torch.mm(center, center.t())    # (200, 200)
            tmp = torch.norm(center, p=2, dim=1, keepdim=True)      # (200, 1)
            norm_matrix = torch.mm(tmp, tmp.t())               # (200, 200)
            cosine_similarity = dot_prod_matrix / norm_matrix  # (200, 200)
            return F.softmax(cosine_similarity, dim=1)
        else:
            raise AssertionError('specified similarity is not implemented, please use euclidean or cosine')

    def forward(self, embedding, logit, label, w):
        batch_size, n_classes = logit.size(0), logit.size(1)
        assert 0 <= w < 1, 'the factor has to be in [0, 1)'

        # smoothed label, shape: (batch size, n_classes)
        smoothed_label = torch.full(size=(batch_size, n_classes), fill_value=self._epsilon / (n_classes - 1))
        smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1-self._epsilon)
        if logit.is_cuda:
            smoothed_label = smoothed_label.cuda()  # (batch size, n_classes)

        # pseudo label
        pseudo_label = self.get_pseudo_label()[label.data]  # (batch size, n_classes)

        # fuse label
        fused_pseudo_label = (1 - w) * smoothed_label + w * pseudo_label
        assert (torch.abs(fused_pseudo_label.sum(dim=1).cpu() - 1) < torch.ones(batch_size) * 1e-5).all(), \
            smoothed_label[0].cpu()

        # calculate cross entropy loss
        log_logit = F.log_softmax(logit, dim=1)
        ce_loss = -torch.sum(log_logit * fused_pseudo_label) / batch_size

        # calculate center loss
        batch_size_tensor = logit.new_empty(1).fill_(batch_size)
        center_loss = self.center_func(embedding, label, self.centers, batch_size_tensor)

        return ce_loss + center_loss * self._lamda


class centerFunc(Function):
    @staticmethod
    def forward(ctx, embeddings, labels, centers, batch_size):
        ctx.save_for_backward(embeddings, labels, centers, batch_size)
        centers_batch = centers.index_select(0, labels.long())
        return (embeddings - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_outputs):
        embeddings, labels, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, labels.long())
        diff = centers_batch - embeddings

        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(labels.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, labels.long(), ones)
        grad_centers.scatter_add_(0, labels.unsqueeze(1).expand(embeddings.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return -grad_outputs * diff / batch_size, None, grad_centers / batch_size, None


def get_init_center(model, loader, n_classes, n_dim_feature=512):
    """
    :param model        : network, should be in cuda
    :param loader       : train loader
    :param n_classes    : number of classes
    :param n_dim_feature: number of feature dimensions
    :return:
        mean_feature: tensor with shape (n_classes, n_dim_feature)
                      class_label_distr[i] is mean feature of class i
    """
    # key: class id, value: logits (num_samples, 512)
    class_feature_dict = {i: torch.zeros((0, n_dim_feature)).cuda() for i in range(n_classes)}
    model.eval()
    with torch.no_grad():
        for it, (x, y) in enumerate(loader):
            x = x.cuda()
            feature_vectors, _ = model(x)
            assert feature_vectors.shape[1] == n_dim_feature
            for i, c in enumerate(y):
                class_feature_dict[c.item()] = torch.cat((class_feature_dict[c.item()],
                                                          feature_vectors[i].unsqueeze(dim=0).detach()))
    mean_feature = torch.zeros(n_classes, n_dim_feature)   # (200, 512)
    for i in range(n_classes):
        mean_feature[i] = class_feature_dict[i].mean(dim=0).cpu()  # (512)

    return mean_feature


if __name__ == '__main__':
    import os
    import sys
    import argparse
    sys.path.append(os.path.abspath('.'))
    from net.vgg import VGG16
    import torchvision
    import torch.utils.data as data
    from torchvision.datasets import ImageFolder
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_data = ImageFolder(os.path.join(args.dataset, 'train'), transform=train_transform)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

    net = VGG16(n_classes=200, pretrained=True)
    net = nn.DataParallel(net).cuda()
    centers_init = get_init_center(net, train_loader, len(train_data.classes), 512)
    torch.save(centers_init, 'centers_init.pt')
