import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import marbleregression.cook_marbleexperiment as cook_regression
from multiprocessing import cpu_count
from torch.nn import functional as F
import torchvision.io
##################################################################
# from here: https://www.kaggle.com/code/purplejester/a-simple-lstm-based-time-series-classifier
##################################################################

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filelist, is_vision=False):
        self.filelist = filelist
        self.is_vision = is_vision
        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        if not self.is_vision:
            id, y, x, x_len, x_mask = pickle.load(open(self.filelist[idx]+".pkl", "rb"))
        else:
            id, y = pickle.load(open(self.filelist[idx]+".pkl", "rb"))
            video_tensor_loaded, _, _ = torchvision.io.read_video(self.filelist[idx]+".mp4", pts_unit="sec")
            # The video tensor will have the shape [T, H, W, C], so we need to permute it to [T, C, H, W]
            #video_tensor_loaded = video_tensor_loaded.permute(0, 3, 1, 2)
            x = video_tensor_loaded.float()

        return x, y
def create_datasets(X, y, test_size=0.2, time_dim_first=False, is_classification=True, dtype_x=torch.float32, device="cpu"):
    if is_classification:
        enc = LabelEncoder()
        y = enc.fit_transform(y)

    X_grouped = X
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_grouped = X_grouped.transpose(1, 0)

    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y, test_size=test_size)
    X_train, X_valid = [torch.tensor(arr, dtype=dtype_x) for arr in (X_train, X_valid)]
    if is_classification:
        y_train, y_valid = [torch.tensor(arr, dtype=torch.long, device=device) for arr in (y_train, y_valid)]
    else:
        y_train, y_valid = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    if is_classification:
        return train_ds, valid_ds, enc
    else:
        return train_ds, valid_ds

def create_dataset_eval(X, y, time_dim_first=False, is_classification = True, device="cpu"):
    if is_classification:
        enc = LabelEncoder()
        y = enc.fit_transform(y)
    X_grouped = X
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_grouped = X_grouped.transpose(1, 0)

    X_valid = torch.tensor(X_grouped, dtype=torch.float32, device=device)
    y_valid = torch.tensor(y, dtype=torch.long, device=device)

    valid_ds = TensorDataset(X_valid, y_valid)
    if is_classification:
        return valid_ds, enc
    else:
        return valid_ds

def create_test_dataset(X):
    X_grouped = torch.tensor(X.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs, pin_memory=True)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs, pin_memory=True)
    return train_dl, valid_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()

def accuracy_on_data(val_dl, model, verbose=False):
    correct, total = 0, 0
    for x_val, y_val in val_dl:
        x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        # print("=====================================================================================================")
        # print(x_val)
        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        # if verbose:
        #     print("Predictions:",preds)
            # print("Ground Trut:", y_val)
        total += y_val.size(0)
        correct += (preds == y_val).sum().detach().item()

    acc = correct / total
    return acc

def mse_on_data(val_dl, model, verbose=False, with_uncertainty=False):
    rmse_all = torch.zeros(0).cuda()
    for x_val, y_val in val_dl:
        x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        # print("=====================================================================================================")
        # print(x_val)
        model.eval()
        if with_uncertainty:
            with torch.no_grad():
                out, out_var = mc_dropout_predict(model, x_val, num_samples=10)
        else:
            out = model(x_val)

        criterion = torch.nn.MSELoss()
        loss = torch.atleast_1d(criterion(y_val, out).detach())
        rmse_all = torch.cat((rmse_all, loss))

        if verbose:
            print("VALIDATION: estimate vs gt", loss, torch.mean(rmse_all))
            print(torch.cat((out.squeeze().unsqueeze(0), y_val.squeeze().unsqueeze(0),), dim=0))

    if with_uncertainty:
        return torch.mean(rmse_all), torch.mean(out_var)
    else:
        return torch.mean(rmse_all)

def mc_dropout_predict(model, inputs, num_samples=10):
    model.dropout.train()  # Enable dropout
    outputs = torch.stack([model(inputs) for _ in range(num_samples)])
    out_mean, out_var = outputs.mean(0), outputs.var(0)
    return out_mean, out_var

def create_eval_dataloader_from_path_2(cooked_dir, modality):
    ds = MyDataset(pickle.load(open(cooked_dir+"_"+modality+"/filelist.pkl", "rb")), is_vision=(modality=="vision"))
    bs = {"ft":16,"sound":16,"vision":4}[modality]

    dl = torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=cpu_count(), pin_memory=True)
    return dl

def create_eval_dataloader_from_path(filepath, modalities_boolean, is_classification, device="cpu"):
    data = pickle.load(open(filepath, "rb"))


    (ids,ys,
     (xs_ft_batch, xs_ft_lengths, xs_ft_mask),
     (xs_sound_batch, xs_sound_lengths, xs_sound_mask),
     (xs_vision_batch, xs_vision_lengths, xs_vision_mask)) = cook_regression.cook_data(data)
    del data
    xs_per_modality = [xs_modality for xs_modality, is_d in
                       zip([xs_ft_batch, xs_sound_batch, xs_vision_batch], modalities_boolean) if is_d]
    del xs_ft_batch, xs_ft_lengths, xs_ft_mask, xs_sound_batch, xs_sound_lengths, xs_sound_mask, xs_vision_batch, xs_vision_lengths, xs_vision_mask



    xs_all_batch = torch.cat(xs_per_modality, dim=2)
    del xs_per_modality
    # construct val ds as also generated by dataloaders

    if is_classification:
        bs = 32
        val_ds, enc = create_dataset_eval(xs_all_batch, ys, is_classification=is_classification, device=device)
    else:
        bs = 4
        val_ds = create_dataset_eval(xs_all_batch, ys, is_classification=is_classification, device=device)
    val_dl = DataLoader(val_ds, bs, shuffle=False, num_workers=cpu_count(), pin_memory=True)
    del val_ds
    return val_dl