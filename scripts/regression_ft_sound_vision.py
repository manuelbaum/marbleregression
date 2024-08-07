import sys
import os
import math
from pathlib import Path

import argparse
import pickle
import torch
import numpy as np
from multiprocessing import cpu_count

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
import marbleregression.cook_marbleexperiment as cook
from marbleregression.util_learning import create_datasets, create_loaders, mse_on_data, create_eval_dataloader_from_path, MyDataset
from marbleregression.models import MLPflat, CNNMLPflat

torch.set_printoptions(precision=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-uid", help="a unique id identifying this run", type=int, default=1000)

    parser.add_argument("-cooked_dir", help="input data directory", default="/media/manuelbaum/Data/marbles/cooked/marbles_heavy")
    parser.add_argument("-out_dir", default="/home/manuelbaum/Data/experiment_marbles/models", help="output directory")
    parser.add_argument("-test_pkl", help="a pickle file for testing out-of distribution performance")#, default="/media/manuelbaum/Data/marbles/raw/marbles_covered/audio_ip.pkl")
    parser.add_argument("--hidden_shared_layerwidth", default=32, help="number of neurons in shared hidden layers", type=int)
    parser.add_argument("--hidden_shared_layercount", default=1, help="number of shared hidden layers", type=int)

    parser.add_argument("--load_pickle", help="directly load from pickle", action="store_false")
    parser.add_argument("--load_cooked", help="directly load from cooked pickle", action="store_true")
    parser.add_argument("--model", help="the model class", choices = ['meanaggregate', 'gaussianaggregate', 'lstm','lstm_simple', 'mlpflat','cnnmlp','cnnmlpgeneric','crossmlpflat','attention','probabilistic','softmoe','combinatorialaggregate'], default="cnnmlpgeneric")
    parser.add_argument("--gpu", help="directly load from pickle", action="store_false")
    parser.add_argument("--modalities", help="the sensor modalities usesd for training. should be any combination of ft,sound,vision and separated by comma (the previous sequence is an example string)", default="vision")
    parser.add_argument("--loss", help="the loss, one of (mse,gaussiannll)", default="mse")

    args = parser.parse_args()

    torch.manual_seed(args.uid)
    np.random.seed(args.uid)

    modalities = args.modalities.split(",")
    is_ft = "ft" in modalities
    is_sound = "sound" in modalities
    is_vision = "vision" in modalities
    modalities_boolean = [is_ft, is_sound, is_vision]


    if is_vision:
        bs = 4
        accumulation_steps = 4  # we do gradient accumulation because we need to have small batch size
        n_epochs = 80
        patience, trials = 100, 0
        lr = 0.001
        max_lr = 1e-2
        base_lr = 1e-5
        weight_decay = 0
        step_size_up = 20
    elif is_sound:
        bs = 24
        accumulation_steps = 1  # we do gradient accumulation because we need to have small batch size
        n_epochs = 2000
        patience, trials = 2000, 0
        lr = 0.0001
        base_lr = 1e-4
        max_lr = 1e-4
        weight_decay = 0
        step_size_up = 100
    elif is_ft:
        bs = 24
        accumulation_steps = 1  # we do gradient accumulation because we need to have small batch size
        n_epochs = 1000
        patience, trials = 1000, 0
        # lr = 0.01
        base_lr = 1e-5
        max_lr = 1e-2
        weight_decay = 1e-1
        step_size_up = 100



    print(args.cooked_dir)
    ds = MyDataset(pickle.load(open(args.cooked_dir+"_"+args.modalities+"/filelist.pkl", "rb")), is_vision=is_vision)
    generator = torch.Generator().manual_seed(args.uid)
    trn_ds, val_ds = torch.utils.data.random_split(ds, [0.8,0.2], generator)

    trn_dl = torch.utils.data.DataLoader(trn_ds, bs, shuffle=False, num_workers=cpu_count(), pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, bs, shuffle=False, num_workers=cpu_count(), pin_memory=True)

    output_dim = 1


    dim_ft = 6
    dim_sound = 129

    dim_mlp = dim_ft * is_ft + dim_sound * is_sound
    dim_vision_flat = 120 * 160 * 3
    dim_vision_post_cnn_flat = x = 1120 #1408 #4800

    if not is_ft and not is_sound and is_vision:
        #cnn
        model_kwargs = {"cnn_flat_dim": dim_vision_post_cnn_flat,  # 129,#6,
                        #"hidden_dim": 32*is_ft+256*is_sound,
                        "hidden_dim_cnn_mlp": 32,
                        "output_dim": output_dim}
        model_hyperparameters = {"class": CNNMLPflat,  # MLPflat,
                                 "kwargs": model_kwargs}
    elif not is_vision:
        #mlp
        model_kwargs = {"input_dim": dim_mlp,  # 129,#6,
                        "hidden_dim": 8*is_ft+16*is_sound,
                        "output_dim": output_dim,
                        "n_batch": bs}
        model_hyperparameters = {"class": MLPflat,  # MLPflat,
                                 "kwargs": model_kwargs}

    model = model_hyperparameters["class"](**model_kwargs)

    model = model.cuda()






    iterations_per_epoch = len(trn_dl)
    best_rmse = sys.float_info.max
    loss_at_best_rmse = sys.float_info.max

    if args.loss == "gaussiannll":
        criterion = nn.GaussianNLLLoss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()# nn.CrossEntropyLoss()

    # opt = torch.optim.RMSprop(model.parameters(), lr=max_lr, weight_decay=1e-1)
    opt = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)#lr=lr)
    # opt = torch.optim.Adadelta(model.parameters())
    # opt = torch.optim.SGD(model.parameters(), lr=max_lr, weight_decay=1e-1)

    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular2')
    # sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)  # Decay LR by a factor of 0.9 every epoch
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)  # Decay LR by a factor of 0.9 every epoch



    # potentially load another dataset to record how out-of-distribution performance would change during learning
    is_test_cross_dataset = args.test_pkl is not None
    if is_test_cross_dataset:
        val_dl_cross = create_eval_dataloader_from_path(args.test_pkl, modalities_boolean, is_classification=False, device="cpu")

        # # test to see what happens when we load data differently (debugging)
        # val_ds_cross, unused_ds, enc = create_datasets(xs_all_batch, ys, test_size = 0.01)
        # val_dl_cross, _ = create_loaders(val_ds_cross, unused_ds, bs, jobs=cpu_count())

    print('Start model training')
    results = {}

    training_dir_name = os.path.basename(args.cooked_dir)
    Path(args.out_dir + "/" + training_dir_name).mkdir(parents=True, exist_ok=True) #make sure the model directory exists (create if necessary)
    output_filepath_prefix = args.out_dir + "/" + training_dir_name + "/" + args.model + "__" + args.modalities + "__" + str(args.uid)

    hist_loss = []
    hist_rmse = []
    hist_rmse_cross = []

    # xs_all_batch = torch.tensor(xs_all_batch).transpose(1, 0).float().cuda()
    # ys = torch.tensor(ys).float().cuda()
    #ys = ys.unsqueeze(1)
    print("====")
    for epoch in range(1, n_epochs + 1):
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            opt.zero_grad()

            out = model(x_batch)
            if args.loss == "gaussiannll":
                out_mean, out_var = out[:,0:1], torch.exp(out[:,1:2])
                loss = criterion(out_mean, y_batch, out_var) + torch.sum(0.01*torch.log(out_var))# torch.sum(0.5*torch.log(2.*3.141*math.e*out_var**2))
            elif args.loss == "mse":
                loss = criterion(out, y_batch)




            # del x_batch, y_batch#outputs, loss
            # torch.cuda.empty_cache()
            # print("computing backward")
            loss.backward()
            if (i + 1) % accumulation_steps == 0:  # Perform optimization every 'accumulation_steps'
                # print("stepping")
                opt.step()
                opt.zero_grad()
            # print("repeat")

        # print("stepping")
        opt.step()
        opt.zero_grad()
        sched.step()

        # model.train()
        #
        # opt.zero_grad()
        # # print("======= computing error ==========")
        # x_batch = xs_all_batch
        # y_batch = ys
        # out = model(x_batch)
        #
        # loss = criterion(out, y_batch)
        #
        # loss.backward()
        # opt.step()
        #
        # print("Epoch:",epoch,"\tLoss:", loss, "\tPred:",out.squeeze())

        model.eval()

        rmse = mse_on_data(val_dl, model)


        if epoch % 1 == 0:
            if is_test_cross_dataset:
                #     # print("======= computing acc_trn ==========")
                #     # acc_trn = accuracy_on_data(trn_dl, model, verbose=True)
                #     # print("======= computing acc_val ==========")
                #     # acc_val = accuracy_on_data(val_dl, model, verbose=True)
                #     # print("======= computing acc_cross ==========")
                rmse_cross = mse_on_data(val_dl_cross, model, verbose=False)
                #     # print("======= computing acc_cross2 ==========")
                #     # acc_cross2 = accuracy_on_data(val_dl_cross2, model, verbose=True)
                hist_rmse_cross.append(rmse_cross)
            if is_test_cross_dataset:
                print(f'Epoch: {epoch:4d}. MSE Train: {loss.item():6.2f}. Valid: {rmse:6.2f}. Transfer:{rmse_cross:6.2f} LR:{sched.get_last_lr()[0]:7.5f}')
            else:
                print(f'Epoch: {epoch:4d}. MSE Train: {loss.item():6.2f}. Valid: {rmse:6.2f} LR:{sched.get_last_lr()[0]:7.5f}')

                # if args.loss == "gaussiannll":
                #     print(torch.cat((out_mean.squeeze().unsqueeze(0), y_batch.squeeze().unsqueeze(0),torch.exp(out_var).squeeze().unsqueeze(0),), dim=0))
                # elif args.loss == "mse":
                #     print(torch.cat((out.squeeze().unsqueeze(0), y_batch.squeeze().unsqueeze(0),), dim=0))

        hist_rmse.append(rmse)

        hist_loss.append(loss.item())

        # if rmse < best_rmse:
        #     # rmse = mse_on_data(val_dl, model, verbose=False, with_uncertainty=args.loss == "gaussiannll")
        #
        #     trials = 0
        #     best_rmse = rmse
        #
        #     results["rmse"]=rmse
        #     results["args"]=args
        #     results["epoch_best"]=epoch
        #
        #     torch.save(model.state_dict(), output_filepath_prefix+'.best.pth')
        #     pickle.dump(model_hyperparameters, open(output_filepath_prefix+'.best_hyper.pkl', "wb"))
        #
        #     # print("estimate vs gt")
        #     # print(torch.cat((out.squeeze().unsqueeze(0),y_batch.squeeze().unsqueeze(0)), dim=0))
        #
        #     if is_test_cross_dataset:
        #         # print(f'Epoch {epoch} best model saved with mse: {best_rmse:.2f}. Loss {loss_at_best_rmse:.2f} Transfer:{rmse_cross:.2f}')
        #         print(f'Epoch: {epoch:4d}. MSE Train: {loss.item():6.2f}. Valid: {rmse:6.2f}. Transfer:{rmse_cross:6.2f} <==== SAVED')
        #     else:
        #         # print(f'Epoch {epoch} best model saved with mse: {best_rmse:.2f}')
        #         print(f'Epoch: {epoch:4d}. MSE Train: {loss.item():6.2f}. Valid: {rmse:6.2f} <==== SAVED')
        #
        # else:
        #     trials += 1
        #     if trials >= patience:
        #         print(f'Early stopping on epoch {epoch}')
        #         break

    ### SAVE LAST MODEL TOO
    rmse = mse_on_data(val_dl, model, verbose=False, with_uncertainty=args.loss == "gaussiannll")
    best_rmse = rmse

    results["rmse"] = rmse
    results["args"] = args
    results["epoch_best"] = epoch

    torch.save(model.state_dict(), output_filepath_prefix + '.best.pth')
    pickle.dump(model_hyperparameters, open(output_filepath_prefix + '.best_hyper.pkl', "wb"))
    ### END SAVE LAST MODEL

    print("Ended with best rmse:",best_rmse,results["rmse"])
    print("Saving to results pickle",output_filepath_prefix+'.results.pkl')
    print("Saved best model to:", output_filepath_prefix + '.best.pth')

    results["hist_rmse"] = hist_rmse
    results["hist_loss"] = hist_loss
    if is_test_cross_dataset:
        results["hist_rmse_cross"] = hist_rmse_cross

    # open a file, where you ant to store the data
    file = open(output_filepath_prefix+'.results.pkl', 'wb')

    # dump information to that file
    pickle.dump(results, file)

if __name__ == "__main__":
    main()