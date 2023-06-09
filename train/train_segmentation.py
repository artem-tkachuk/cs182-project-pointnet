from __future__ import print_function
import os
import random
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from cs182_project_pointnet.utils.options import Options
from cs182_project_pointnet.dataset.ShapeNetDataset import ShapeNetDataset
from cs182_project_pointnet.pointnet.models import PointNetDenseCls, feature_transform_regularizer


def train_segmentation(
        dataset,
        workers=4,
        nepoch=25,
        outf='seg',
        model='',
        class_choice='Chair',
        feature_transform=False
):
    ############# YOUR CODE HERE ###############
    #  Tune hyperparameters                    #
    ############################################
    lr = ?
    step_size = ?
    batch_size = ?
    ############# END YOUR CODE ###############

    opt = Options(
        dataset=dataset,
        batchSize=batch_size,
        workers=workers,
        nepoch=nepoch,
        outf=outf,
        model=model,
        class_choice=class_choice,
        feature_transform=feature_transform
    )


    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            # print(pred.size(), target.size())
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * 2500)))

            ############# YOUR CODE HERE ####################
            #  Store train loss and accuracy for downstream #
            #  visualizations                               #
            #  Hint: append to lists previously defined     #
            #################################################
            # TODO: a line here
            train_accuracies.append(? / float(opt.batchSize * 2500))
            ############# END YOUR CODE #####################

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize * 2500)))

                # For test
                test_losses.append(loss.item())
                test_accuracies.append(correct.item() / float(opt.batchSize * 2500))

        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

    ## benchmark mIOU
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))

    ##### VISUALIZE TRAINING CURVES ####
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot train accuracy
    axes[0, 0].plot(train_accuracies)
    axes[0, 0].set_title('Train Accuracy')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Accuracy')

    # Plot test accuracy
    axes[0, 1].plot(test_accuracies)
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Accuracy')

    # Plot train loss
    axes[1, 0].plot(train_losses)
    axes[1, 0].set_title('Train Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')

    # Plot test loss
    ############# YOUR CODE HERE ####################
    #  Plot the test loss                           #
    #  Hint: Look at the code above for inspiration #
    #################################################
    # TODO: code here
    ############# END YOUR CODE #####################

    # Show the plots
    plt.tight_layout()
    plt.show()

