from sklearn.metrics import roc_curve, auc
from paddle.vision import transforms
from paddle.io import DataLoader
import paddle
from dataset import MVTecAT
from model import ProjectionNet
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
from density import GaussianDensityPaddle
import pandas as pd
import sys
import os


parser = argparse.ArgumentParser(description='eval models')
parser.add_argument('--type', default="bottle,grid",
                    help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

parser.add_argument('--pretrained', default="models_param",
                    help=' directory contating models to evaluate (default: models)')

parser.add_argument('--data_dir', default="lite_data",
                    help=' directory contating data to evaluate')

parser.add_argument('--logs_dir', default="logs",
                    help='logs folder of the models ')

parser.add_argument('--cuda', default=True,
                    help='use cuda for model predictions (default: False)')

parser.add_argument('--head_layer', default=1, type=int,
                    help='number of layers in the projection head (default: 8)')

parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

parser.add_argument('--save_plots', default=True,
                    help='save TSNE and roc plots')
parser.add_argument('--seed', default=1012, type=int, help="number of random seed")
args = parser.parse_args()


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def reset(self):
        self.log.close()
        sys.stdout = self.terminal
    def flush(self):
        pass


def eval_model(data_dir="../images", pretrained=None, data_type=None, size=256,
               head_layer=8, density=GaussianDensityPaddle()):
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    test_data_eval = MVTecAT(data_dir, data_type, size, transform=test_transform, mode="test")
    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                 shuffle=False, num_workers=0)
    model_name = pretrained+'/model_'+data_type
    print(f"loading model {model_name}")
    head_layers = [512] * head_layer + [128]
    print(head_layers)
    weights = paddle.load(model_name + '.pdparams')
    num_classes = 3
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    model.set_state_dict(weights)
    model.eval()

    labels = []
    embeds = []
    with paddle.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x)
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = paddle.concat(labels)
    embeds = paddle.concat(embeds)
    embeds = paddle.nn.functional.normalize(embeds, p=2, axis=1)
    density.load(f'{model_name}_dict_data.pkl')
    distances = density.predict(embeds)
    fpr, tpr, _ = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    return roc_auc


if __name__ == '__main__':
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(f'{args.logs_dir}'):
        os.makedirs(f'{args.logs_dir}')
    logger = Logger(f"{args.logs_dir}/eval_process.txt")

    paddle.device.set_device("cpu" if args.cuda in ["False","no","false",'0',0,False] else "gpu:0")
    print(args)
    all_types = ['bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper']

    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")

    density_mapping = {
        "paddle": GaussianDensityPaddle,
    }

    density = density_mapping[args.density]

    # find models
    model_names = [Path(args.pretrained) / f"model_{data_type}" for data_type in types if
                   len(list(Path(args.pretrained).glob(f"model_{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    obj = defaultdict(list)
    all_x = []
    eval_dir = Path("eval") / args.pretrained
    eval_dir.mkdir(parents=True, exist_ok=True)

    for model_name, data_type in zip(model_names, types):
        print(f"evaluating {data_type}")

        roc_auc = eval_model(data_dir=args.data_dir, pretrained=args.pretrained, data_type=data_type,
                             head_layer=args.head_layer, density=density())

        print(f"{data_type} AUC: {roc_auc}")
        obj["data_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)
        all_x.append(roc_auc)

    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "_perf.csv")
    print('平均结果：'+str(np.mean(all_x)))
    logger.reset()
