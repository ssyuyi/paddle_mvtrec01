from pathlib import Path
import argparse
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.vision import transforms
import numpy as np
import time
from dataset import MVTecAT
from cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model import ProjectionNet
from sklearn.metrics import roc_curve, auc
from density import GaussianDensityPaddle
import pickle
import sys
import os

parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
parser.add_argument('--type', default='bottle,grid,hazelnut,leather,metal_nut,pill',
                    help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

parser.add_argument('--epochs', default=256, type=int,
                    help='number of epochs to train the model , (default: 256)')

parser.add_argument('--data_dir', default="images",
                    help='input folder of the models ,')

parser.add_argument('--model_dir', default="modelsx_param",
                    help='output folder of the models , (default: models)')

parser.add_argument('--logs_dir', default="logs3",
                    help='logs folder of the models ')

parser.add_argument('--cuda', default=True,
                    help='use cuda for training')

parser.add_argument('--no_pretrained', default=True, action='store_false',
                    help='use pretrained values to initalize ResNet18 , (default: True)')

parser.add_argument('--test_epochs', default=1, type=int,
                    help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')

parser.add_argument('--freeze_resnet', default=20, type=int,
                    help='number of epochs to freeze resnet (default: 20)')

parser.add_argument('--lr', default=0.03, type=float,
                    help='learning rate (default: 0.03)')

parser.add_argument('--optim', default="sgd",
                    help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

parser.add_argument('--batch_size', default=96, type=int,
                    help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')

parser.add_argument('--head_layer', default=1, type=int,
                    help='number of layers in the projection head (default: 1)')

parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'],
                    help='cutpaste variant to use (dafault: "3way")')

parser.add_argument('--workers', default=1, type=int, help="number of workers to use for data loading (default:8)")
parser.add_argument('--seed', default=1012, type=int, help="number of random seed")

args = parser.parse_args()



class Logger(object):
    def __init__(self, filename="Default.txt"):
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


def run_training(
        data_type,
        model_dir,
        data_dir="images",
        epochs=256,
        pretrained=True,
        test_epochs=10,
        freeze_resnet=20,
        learninig_rate=0.03,
        batch_size=64,
        head_layer=8,
        cutpate_type=CutPasteNormal,
        workers=1,
        size=256,
        print_freq = 1,
        test_batch_size=64
):

    weight_decay = 0.00003
    momentum = 0.9
    min_scale = 1
    model_name = f"model_{data_type}"
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size, size)))
    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))
    train_datas = MVTecAT(data_dir, data_type, transform=train_transform, size=int(size * (1 / min_scale)))
    train_loader = DataLoader(train_datas, batch_size=batch_size, drop_last=False,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True)


    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    test_datas = MVTecAT(data_dir, data_type, size, transform=test_transform, mode="test")
    test_loader = DataLoader(test_datas, batch_size=test_batch_size,
                                 shuffle=False, num_workers=0)
    train_datas_for_test = MVTecAT(data_dir, data_type, size, transform=test_transform)
    train_loader_for_test = DataLoader(train_datas_for_test, batch_size=test_batch_size,
                                       shuffle=False, num_workers=0)

    head_layers = [512] * head_layer + [128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = paddle.nn.CrossEntropyLoss()

    scheduler = CosineAnnealingDecay(learning_rate=learninig_rate, T_max=epochs//2)
    optimizer = optim.Momentum(learning_rate=scheduler, momentum=momentum, parameters=model.parameters(), use_nesterov=True, weight_decay=weight_decay)

    max_roc = -np.inf
    model.train()

    for epoch in range(epochs):
        # training log
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        batch_past = 0

        if epoch == freeze_resnet:
            model.unfreeze()

        for train_idx, train_data in enumerate(train_loader):

            train_reader_cost += time.time() - reader_start
            train_start = time.time()

            optimizer.clear_grad()

            datax = paddle.concat(train_data, axis=0)
            embeds, logits = model(datax)
            y = paddle.arange(len(train_data))
            y = paddle.repeat_interleave(y, train_data[0].shape[0])
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step(epoch)
            train_run_cost += time.time() - train_start
            total_samples += train_data[0].shape[0]
            batch_past += 1

            if print_freq > 0 and (train_idx+1) % print_freq == 0:
                msg = "Type : {} Train [ Epoch {}/{} iter {} ] lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                    data_type, epoch+1, epochs,train_idx+1,
                    optimizer.get_lr(),
                    loss.item(), train_reader_cost / batch_past,
                                 (train_reader_cost + train_run_cost) / batch_past,
                                 total_samples / batch_past,
                                 total_samples / (train_reader_cost + train_run_cost))
                print(msg)
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                batch_past = 0

            reader_start = time.time()

        if test_epochs > 0 and (epoch + 1) % test_epochs == 0:
            test_start_time = time.time()
            model.eval()
            train_embeds = []
            with paddle.no_grad():
                density = GaussianDensityPaddle()
                for x in train_loader_for_test:
                    embed, _ = model(x)
                    train_embeds.append(embed)
                train_embeds = paddle.concat(train_embeds)
                train_embeds = paddle.nn.functional.normalize(train_embeds, p=2, axis=1)
                density.fit(train_embeds)
                test_embeds = []
                test_labels = []
                for test_data, labels in test_loader:
                    embeds, logits = model(test_data)
                    test_embeds.append(embeds)
                    test_labels.append(labels)
                test_labels = paddle.concat(test_labels)
                test_embeds = paddle.concat(test_embeds)
                test_embeds = paddle.nn.functional.normalize(test_embeds, p=2, axis=1)

                distances = density.predict(test_embeds.cpu())
                fpr, tpr, threshold = roc_curve(test_labels, distances)
                right_index = (tpr + (1 - fpr) - 1)
                index = np.argmax(right_index)
                best_threshold = threshold[index]
                dict_data = {
                    'inv_cov': density.inv_cov.numpy(),
                    'mean': density.mean.numpy(),
                    'best_threshold': best_threshold
                }
                fo =  open(str(model_dir / f"{model_name}_dict_data.pkl"), 'wb')
                pickle.dump(dict_data, fo)
                fo.close()
                roc_auc = auc(fpr, tpr)

            model.train()

            test_end_time = time.time()
            if max_roc <= roc_auc:
                max_roc = roc_auc
                paddle.save(model.state_dict(), str(model_dir / f"{model_name}.pdparams"))
            msg = "Type : {} Test [ Epoch {}/{} ] test_cost: {:5f} sec max_auc: {:.5f} roc_auc : {:5f}.".format(
                data_type, epoch+1, epochs, test_end_time - test_start_time, max_roc, roc_auc)
            print(msg)


if __name__ == '__main__':
    paddle.set_device("cpu" if args.cuda in ["False","no","false",'0',0,False] else "gpu")

    paddle.seed(args.seed)
    np.random.seed(args.seed)

    print(args)
    all_types = ['bottle',       # 1
                 'cable',        # 0.92466
                 'capsule',      # 0.92859
                 'carpet',       # 0.89767
                 'grid',         # 0.99916
                 'hazelnut',     # 0.96571
                 'leather',      # 1
                 'metal_nut',    # 0.96041
                 'pill',         # 0.95499
                 'screw',        # 0.86288
                 'tile',         # 0.98954
                 'toothbrush',   # 0.99167
                 'transistor',   # 0.95625
                 'wood',         # 0.98772
                 'zipper']       # 1
                                 # avg 0.96128
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")

    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion}
    variant = variant_map[args.variant]

    Path(args.model_dir).mkdir(exist_ok=True, parents=True)

    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for data_type in types:

        if not os.path.exists(f'{args.logs_dir}/{data_type}'):
            os.makedirs(f'{args.logs_dir}/{data_type}')
        logger = Logger(f"{args.logs_dir}/{data_type}/train_process.txt")
        print(f"\n############################\ntraining {data_type}\n############################\n")
        run_training(
            data_type,
            data_dir=args.data_dir,
            model_dir=Path(args.model_dir),
            epochs=args.epochs,
            pretrained=args.no_pretrained,
            test_epochs=args.test_epochs,
            freeze_resnet=args.freeze_resnet,
            learninig_rate=args.lr,
            batch_size=args.batch_size,
            head_layer=args.head_layer,
            cutpate_type=variant,
            workers=args.workers)
        logger.reset()
