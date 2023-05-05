import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from cs182_project_pointnet.dataset.ShapeNetDataset import ShapeNetDataset
from cs182_project_pointnet.pointnet.models import PointNetDenseCls
from cs182_project_pointnet.utils.options import Options
from cs182_project_pointnet.visualize.critical_points import compute_principal_curvature, visualize_critical_points
from cs182_project_pointnet.visualize.show_points import show_points


def visualize_seg_and_critical_points(
        model,
        show,  # what to show: 'prediction' or 'gt' or 'critical_points'
        idx=0,
        dataset='',
        class_choice='',
):
    assert show in ['predictions', 'gt', 'critical_points'], "Unknown show option"

    opt = Options(
        model=model,
        idx=idx,
        dataset=dataset,
        class_choice=class_choice
    )

    # print(opt)

    d = ShapeNetDataset(
        root=opt.dataset,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)

    idx = opt.idx

    # print("model %d/%d" % (idx, len(d)))
    point, seg = d[idx]
    # print(point.size(), seg.size())
    point_np = point.numpy()

    cmap = plt.cm.get_cmap("hsv", 10);
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    gt = cmap[seg.numpy() - 1, :]

    state_dict = torch.load(opt.model)
    classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0])
    classifier.load_state_dict(state_dict)
    classifier.eval()

    point = point.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]
    # print(pred_choice)

    # print(pred_choice.size())
    pred_color = cmap[pred_choice.numpy()[0], :]

    # print(pred_color.shape)
    # print(len(d))

    if show == 'predictions':
        show_points(point_np, pred_color, title='Prediction')

    elif show == 'gt':
        ############ YOUR CODE HERE #####################
        #  Show ground truth visualization              #
        #  visualizations                               #
        #################################################
        show_points(?, ?, title='Ground Truth')
        ############# END YOUR CODE #####################

    elif show == 'critical_points':
        curvatures = compute_principal_curvature(point_np)
        print('Critical points:')
        visualize_critical_points(
            point_np,
            pred_choice.numpy()[0],
            curvatures,
            curvature_threshold=0.1
        )