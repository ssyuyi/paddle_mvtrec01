from paddle.vision import transforms
from paddle.io import DataLoader
import paddle
from dataset import MVTecAT
from model import ProjectionNet
from PIL import Image
from density import GaussianDensityPaddle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='use this model.')
parser.add_argument('--cuda', default=True, help='use gpu')
parser.add_argument('--data_type', default='bottle', type=str, help='which data type need to be detected')
parser.add_argument('--img_file', default='demo/bottle_good.png', type=str, help='the path of image that will be detected')
parser.add_argument('--model_params', default='models_param/', type=str,
                    help='where the params of model saved')
parser.add_argument('--seed', default=1012, type=int, help="number of random seed")
args = parser.parse_args()



def predict_img(img_file, model, density, transform, size=256):
    img = Image.open(img_file).resize((size, size)).convert("RGB")
    img = transform(img)
    img = paddle.unsqueeze(img, axis=0)
    with paddle.no_grad():
        embed, logit = model(img)
    embed = paddle.nn.functional.normalize(embed, p=2, axis=1)
    res = density.predict(embed)
    return res


def load_all_model(model_name, size=256,
               head_layer=1, density=GaussianDensityPaddle(), num_classes=3):
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    model_name_real = str(model_name) + '.pdparams'
    weights = paddle.load(model_name_real)
    head_layers = [512] * head_layer + [128]
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    model.set_state_dict(weights)
    model.eval()

    density.load(str(model_name)+'_dict_data.pkl')
    return model, density, test_transform


def infer(model_name, img_file):
    model, density, trans = load_all_model(model_name)
    in_ = img_file
    res = predict_img(in_, model, density, trans)
    print(f'预测分数：{res.numpy()[0]}{f" > 最佳阈值：{density.best_threshold.numpy()[0]}" if res > density.best_threshold else f" < 最佳阈值： {density.best_threshold.numpy()[0]}"} ')
    print(f'预测结果：{"异常数据" if res > density.best_threshold else "正常数据"}')

if __name__=='__main__':
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    paddle.device.set_device("cpu" if args.cuda in ["False","no","false",'0',0,False] else "gpu:0")
    model_name = args.model_params+"model_"+args.data_type
    infer(model_name, args.img_file)