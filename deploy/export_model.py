import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import pickle
import sys
import numpy as np
from tools.density import GaussianDensityPaddle
paddle.set_device("cpu")
from tools.model import ProjectionNet

def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--model_path', default='models_param', help='model_weight')
    parser.add_argument('--headlayer', default=1,type=int, help='headlayer for the model')
    parser.add_argument('--data_type', default='bottle', help='defect type for the model')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--img_size', default=256, help='image size to export')
    parser.add_argument(
        '--save_inference_dir', default='deploy', help='path where to save')
    parser.add_argument('--pretrained', default=None, help='pretrained model')
    parser.add_argument('--num_classes', default=3,type=int, help='num_classes')

    args = parser.parse_args()
    return args


def export(args):
    # model = paddlevision.models.__dict__[args.model](
    #     pretrained=args.pretrained, num_classes=args.num_classes)
    head_layers = [512] * args.headlayer + [128]
    model = ProjectionNet(pretrained=args.pretrained, head_layers=head_layers, num_classes=args.num_classes)
    # model = nn.Sequential(model, nn.Softmax())
    model.eval()
    # print("%s/%s/final.pdparmas"%(args.model_path,args.data_type))
    model_dict = paddle.load("%s/model_%s.pdparams"%(args.model_path,args.data_type))
    model.set_dict(model_dict)

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))

    density = GaussianDensityPaddle()
    density.load(args.model_path+'/model_'+args.data_type + '_dict_data.pkl')
    dict_data = {
        'inv_cov': density.inv_cov.numpy(),
        'mean': density.mean.numpy(),
        'best_threshold': density.best_threshold.numpy()
    }
    with open(args.save_inference_dir + 'model_'+args.data_type + f'_dict_data.pkl', 'wb') as fo:
        pickle.dump(dict_data, fo)

    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)