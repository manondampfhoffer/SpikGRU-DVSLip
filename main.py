"""
To train and test the model
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans

import DVSLip
import models

parser = argparse.ArgumentParser()    
parser.add_argument('-f', dest='filename', default='test', type=str, help='filename to store the model')
parser.add_argument('-t', dest='is_test', action='store_true', default=False, help='test only')
parser.add_argument('-e', dest='epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('-a', dest='is_ann', action='store_true', default=False, help='ann network')
parser.add_argument('--actreg', default=0.0, type=float, help='activity regularization for SNNs')
parser.add_argument('--finetune', action='store_true', default=False, help='restart training from the given model')
parser.add_argument('--nbframes', default=30, type=int, help='nb of frames for data pre-processing')
parser.add_argument('-b', dest='batch_size', default=32, type=int, help='training batch_size')
parser.add_argument('--augS', action='store_true', default=False, help='spatial data augmentation (for training)')
parser.add_argument('--augT', action='store_true', default=False, help='temporal data augmentation (for training)')
parser.add_argument('--ternact', dest='ternact', action='store_true', default=False, help='SNN with ternary activations')
parser.add_argument('--useBN', dest='useBN', action='store_true', default=False, help='use batch norm in Conv layers')
parser.add_argument('--front', action='store_true', default=False, help='train front end (resnet) only')
parser.add_argument('--NObidirectional', action='store_true', default=False, help='NO bidirectional GRU')
parser.add_argument('--singlegate', action='store_true', default=False, help='NO two gates GRU (single gate)')
parser.add_argument('--hybridsign', action='store_true', default=False, help='hybrid signed SNN : bin frontend / tern backend')
parser.add_argument('--hybridANN', action='store_true', default=False, help='hybrid ANN-SNN : SNN frontend / ANN backend')
parser.add_argument('--nowarmup', action='store_true', default=False, help='no warmup epoch')
parser.add_argument('--Tnbmask', default=6, type=int, help='nb of masks for temporal data augmentation')
parser.add_argument('--Tmaxmasklength', default=18, type=int, help='maximale length of each mask for temporal data augmentation')

args = parser.parse_args()
device = torch.device("cuda:0")
dtype = torch.float
SAVE_PATH_MODEL_BEST = os.getcwd() + '/' + args.filename + '.pt'

## DATASET
####################################################################
train_data_root = 'DVSLIP/train'
test_data_root = 'DVSLIP/test'
training_words = DVSLip.get_training_words()
label_dct = {k:i for i,k in enumerate(training_words)}
## label_dct: {'education': 0, 'between': 1, 'london': 2, 'allow': 3, 'military': 4, 'warning': 5, 'little': 6, 'press': 7, 'missing': 8, 'numbers': 9, 'change': 10, 'support': 11, 'immigration': 12, 'started': 13, 'still': 14, 'attacks': 15, 'called': 16, 'another': 17, 'security': 18, 'minutes': 19, 'point': 20, 'general': 21, 'judge': 22, 'hundreds': 23, 'spend': 24, 'described': 25, 'million': 26, 'having': 27, 'young': 28, 'syria': 29, 'evening': 30, 'american': 31, 'difference': 32, 'russian': 33, 'taken': 34, 'potential': 35, 'russia': 36, 'terms': 37, 'banks': 38, 'leaders': 39, 'welcome': 40, 'house': 41, 'labour': 42, 'words': 43, 'challenge': 44, 'taking': 45, 'worst': 46, 'everything': 47, 'really': 48, 'needs': 49, 'america': 50, 'allowed': 51, 'under': 52, 'thing': 53, 'happened': 54, 'price': 55, 'syrian': 56, 'benefit': 57, 'paying': 58, 'right': 59, 'tomorrow': 60, 'capital': 61, 'question': 62, 'germany': 63, 'meeting': 64, 'these': 65, 'couple': 66, 'saying': 67, 'billion': 68, 'majority': 69, 'think': 70, 'accused': 71, 'giving': 72, 'action': 73, 'become': 74, 'economic': 75, 'times': 76, 'different': 77, 'perhaps': 78, 'benefits': 79, 'court': 80, 'water': 81, 'death': 82, 'during': 83, 'chief': 84, 'happen': 85, 'being': 86, 'years': 87, 'election': 88, 'ground': 89, 'england': 90, 'exactly': 91, 'should': 92, 'spent': 93, 'several': 94, 'number': 95, 'around': 96, 'significant': 97, 'legal': 98, 'heavy': 99}

train_dataset = DVSLip.DVSLipDataset(train_data_root, label_dct, train=True, augment_spatial=args.augS, augment_temporal=args.augT, T=args.nbframes, Tnbmask=args.Tnbmask, Tmaxmasklength=args.Tmaxmasklength)
test_dataset = DVSLip.DVSLipDataset(test_data_root, label_dct, train=False, augment_spatial=False, augment_temporal=False, T=args.nbframes)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True) 
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True) 
# print(len(train_dataloader), len(test_dataloader)) # 14896 train / 4975 test so 466 / 156 iter with bs 32


### MODEL ##
###############################################################
model = models.SCNN(args, 100, useBN=args.useBN, ternact=args.ternact, ann=args.is_ann, front=args.front, NObidirectional=args.NObidirectional, singlegate=args.singlegate, hybridsign=args.hybridsign, hybridANN=args.hybridANN)
model.cuda()

print(model)
param_flatten = torch.cat([param.data.view(-1) for param in model.parameters()], 0)
print("nb param:", param_flatten.size())
print(args)


def train(model, loss_fn, optimizer, train_dataloader, valid_dataloader, nb_epochs, scheduler=None, warmup_epochs=0):
    """ 
    Train the model
    """
    if warmup_epochs > 0:
        for g in optimizer.param_groups:
            g['lr'] /= len(train_dataloader)*warmup_epochs
        warmup_itr = 1
    best_val = 0; best_epoch = 0

    for e in range(nb_epochs):
        local_loss = []
        train_accs = []
            
        for ni, (x_batch, y_batch) in enumerate(train_dataloader):
            model.train()
            x_batch = x_batch.to(device, dtype, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            output, loss_act, spike_act = model(x_batch)
            
            log_p_y = torch.mean(output, dim=1)
            loss_val = loss_fn(log_p_y, y_batch) 

            if args.actreg > 0:
                loss_val += args.actreg * loss_act

            am = torch.argmax(log_p_y, dim=1)
            tmp = np.mean((y_batch==am).detach().cpu().numpy())
            train_accs.append(tmp)

            local_loss.append(loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()

            if ni % 100 == 0:
                print("iter %i loss %.4f loss_act %.4f lr: %.5f"%(ni, loss_val.item(), loss_act.item(), optimizer.param_groups[0]["lr"]))
                # if not args.is_ann:
                #     for i in range(len(spike_act)):
                #         print(spike_act[i], end = ' ')
                #     print("")

            optimizer.step()

            model.clamp()

            if e < warmup_epochs:
                for g in optimizer.param_groups:
                    g['lr'] *= (warmup_itr+1)/(warmup_itr)
                warmup_itr += 1


        mean_loss = np.mean(local_loss)
        train_accuracy = np.mean(train_accs)
        print("Epoch %i: loss=%.5f, Training accuracy=%.3f"%(e+1, mean_loss, train_accuracy))
        
        valid_accuracy = compute_classification_accuracy(model, valid_dataloader, valid=True)
        print("Validation accuracy=%.3f"%(valid_accuracy))

        if scheduler is not None and e >= warmup_epochs:
            scheduler.step()
        
        if valid_accuracy > best_val:
            print("better valid_accuracy")
            torch.save(model.state_dict(), SAVE_PATH_MODEL_BEST)
            best_val = valid_accuracy
            best_epoch = e

        with open('res_' + args.filename + '.txt', 'a') as f:
            f.write("epoch %i: train: %.2f, val: %.2f, loss: %.5f, loss_reg: %.3f, lr: %.5f\n"%(e+1, train_accuracy*100, valid_accuracy*100, mean_loss, loss_act, optimizer.param_groups[0]["lr"]))
                 

    with open('res_' + args.filename + '.txt', 'a') as f:
        f.write("best epoch, accu (val): %i %.2f"%(best_epoch +1, best_val*100))
        f.write('\n')
    return

def compute_classification_accuracy(model, dataloader, valid=False):
    """ 
    Evaluate the model on the given dataset (accuracy and spike rate). 
    If valid=True, do not compute the spike rate. 
    """
    accs = np.array([])
    avg_spike_act = None
    avg_in_act = []

    model.eval()
    with torch.no_grad():
        for ni, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device, dtype, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            output, loss_act, spike_act = model(x_batch)

            log_p_y = torch.mean(output, dim=1)

            am = torch.argmax(log_p_y, dim=1)
            tmp = (y_batch==am).view(-1).detach().cpu().numpy()
            accs = np.concatenate((accs, tmp))

            if avg_spike_act == None:
                avg_spike_act = [ [] for i in range(len(spike_act))]
            for i,l in enumerate(spike_act):
                avg_spike_act[i].append(l)
            avg_in_act.append(x_batch.abs().mean().detach().cpu().numpy())

            if ni % 20 == 0:
                print("testing iter...", ni, np.mean(accs))

    testaccu = np.mean(accs)
    print("test accu:", testaccu)

    if len(spike_act) == 22:
        layer_names = ["conv1", "pool1", "conv2_1", "conv2_11", "conv2_2", "conv2_21", "conv3_1", "conv3_11", "conv3_2", "conv3_21", "conv4_1", "conv4_11", "conv4_2","conv4_21","conv5_1","conv5_11","conv5_2", "conv5_21","avgpool","gru1", "gru2", "gru3"]
    else:
        layer_names = ["conv1", "pool1", "conv2_1", "conv2_11", "conv2_2", "conv2_21", "conv3_1", "conv3_11", "conv3_2", "conv3_21", "conv4_1", "conv4_11", "conv4_2","conv4_21","conv5_1","conv5_11","conv5_2", "conv5_21","avgpool"]
    if not args.is_test and not valid:
            with open('res_' + args.filename + '.txt', 'a') as f:
                f.write("INPUT activity: %0.4f"%(np.mean(avg_in_act)))
                for i,l in enumerate(avg_spike_act):
                    f.write("avg spike activity %s: %0.4f \n"%(layer_names[i], np.mean(avg_spike_act[i])))
                f.write("# avg spike activity ALL: %0.4f \n"%(np.mean(avg_spike_act)))
    else:
        print("INPUT activity: %0.4f"%(np.mean(avg_in_act)))
        for i,l in enumerate(avg_spike_act):
            print("avg spike activity %s: %0.4f"%(layer_names[i], np.mean(avg_spike_act[i])))
        print("# avg spike activity ALL: %0.4f"%(np.mean(avg_spike_act)))

    return testaccu



## TRAINING PARAMETERS
########################################################################
if not args.is_test:
    print("training filename:", args.filename)
    print("training iter:", len(train_dataloader))
    if args.finetune:
        model.load_state_dict(torch.load(SAVE_PATH_MODEL_BEST), strict=False) # strict=False ignores the unmatching keys in both state_dict
        print("FINE TUNE: ######## FILE LOADED:", SAVE_PATH_MODEL_BEST)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    if args.is_ann:
        if args.finetune:
            lr = 1e-4 * (args.batch_size / 32) 
        else:
            lr = 3e-4 * (args.batch_size / 32)
        final_lr = 5e-6 * (args.batch_size / 32)
    else:
        if args.finetune:
            lr = 1e-4 * (args.batch_size / 32)
            final_lr = 5e-6 * (args.batch_size / 32)
        else:
            lr = 3e-4 * (args.batch_size / 32)
            final_lr = lr

    if args.nowarmup:
        warmup_epochs = 0
    else:
        warmup_epochs = 1

    params = []
    for name, param in model.named_parameters():
        if "bn" in name: # no weight decay on batch norm param
            params += [{'params':param, 'lr':lr}]
        else:
            params += [{'params':param, 'lr': lr, 'weight_decay':1e-4}]
    optimizer = torch.optim.Adam(params)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=final_lr, last_epoch=-1)
    
    train(model, loss_fn, optimizer, train_dataloader, test_dataloader, args.epochs, scheduler, warmup_epochs)


## LOAD MODEL AND FINAL TEST
######################################
model.load_state_dict(torch.load(SAVE_PATH_MODEL_BEST), strict=True)
print("######## FILE LOADED:", SAVE_PATH_MODEL_BEST)

test_accuracy = compute_classification_accuracy(model, test_dataloader)
print("test accuracy:" + str(test_accuracy) + "\n")

if not args.is_test:
    with open('res_' + args.filename + '.txt', 'a') as f:
        f.write("\nTest accuracy(noerror) %s \n"%(test_accuracy))