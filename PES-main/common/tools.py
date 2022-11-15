import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import visdom

def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, train_loader, optimizer, ceriation, epoch,flag):
    vis = visdom.Visdom(port=6006)
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    with tqdm(total=len(train_loader),desc="Train",leave=True) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            logist = model(images)
            loss = ceriation(i,logist, labels)

            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            pbar.update(1)
            end = time.time()
    if flag==1:
        vis.line([losses.avg], [epoch], win="train", update="append", name="loss")
        vis.line([top1.avg.to("cpu", torch.float).item()], [epoch], win="train", update="append", name="acc")
    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item()


def evaluate(model, eva_loader, ceriation, prefix,epoch,flag,ignore=-1):
    losses1 = AverageMeter('Loss', ':3.2f')
    top11 = AverageMeter('Acc@1', ':3.2f')
    model.eval()
    vis=visdom.Visdom(port=6006)
    with tqdm(total=len(eva_loader),leave=True,desc="Evaluate") as pbar:
        with torch.no_grad():
            for i, (images, labels) in enumerate(eva_loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()

                logist = model(images)

                loss = F.cross_entropy(logist, labels)
                acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

                losses1.update(loss.item(), images[0].size(0))
                top11.update(acc1[0], images[0].size(0))
                pbar.update(1)
    if flag==1:
        vis.line([losses1.avg], [epoch], win="test", update="append", name="loss")
        vis.line([top11.avg.to("cpu", torch.float).item()], [epoch], win="test", update="append", name="acc")
    if prefix != "":
        print(getTime(), prefix, round(top11.avg.item(), 2))

    return losses1.avg, top11.avg.to("cpu", torch.float).item()


def evaluateWithBoth(model1, model2, eva_loader, prefix):
    top1 = AverageMeter('Acc@1', ':3.2f')
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist1 = model1(images)
            logist2 = model2(images)
            logist = (F.softmax(logist1, dim=1) + F.softmax(logist2, dim=1)) / 2
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return top1.avg.to("cpu", torch.float).item()


def predict(predict_loader, model):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _ in predict_loader:
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model):

    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1 = model(images1)
                logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()

import torch.nn as nn


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lambda1 = 3, beta=0.7,batch_size=128):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta
        self.lambda1 = lambda1
        self.batch_size=batch_size

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        for i in range(0,self.batch_size):
            self.target[index*self.batch_size+i] = self.beta * self.target[index*self.batch_size+i] + (1 - self.beta) * (
                        (y_pred_)[i] / (y_pred_)[i].sum(dim=0))
        self.target1=self.target[index*self.batch_size:(index+1)*self.batch_size]
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target1 * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lambda1 *elr_reg
        return final_loss

