import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, LayerNorm

from torch import optim
from torch.optim.lr_scheduler import StepLR

import time

device = torch.device('cuda:0')

# dataset

training_data = torch.load('./data/AFPDB_data_128_Train_small.pt')
validation_data = torch.load('./data/PDB_data_128_Val_complete.pt')

length = 128

train_loader = DataLoader(training_data, batch_size=16, shuffle=False, num_workers=0)
valid_loader = DataLoader(validation_data, batch_size=16, shuffle=False, num_workers=0)

from model import ProteinAE

model = ProteinAE(device=device).double()
num_params = sum(p.numel() for p in model.parameters())

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

reg_criterion = torch.nn.L1Loss()
multi_class_criterion = torch.nn.CrossEntropyLoss()
binary_class_criterion = torch.nn.BCELoss()

def torsion_angle(coords_ca):

    v1 = coords_ca[1:-2] - coords_ca[0:-3]  # r_ji
    v2 = coords_ca[2:-1] - coords_ca[1:-2]  # r_kj
    v3 = coords_ca[3:] - coords_ca[2:-1]  # r_lk
    v1 = v1 / v1.norm(dim=1, keepdim=True)
    v2 = v2 / v2.norm(dim=1, keepdim=True)
    v3 = v3 / v3.norm(dim=1, keepdim=True)
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)
    a = (n1 * n2).sum(dim=-1)
    b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))

    torsion_angle = torch.nan_to_num(torch.atan2(b, a))

    return torsion_angle

edgeloss_weight=0.5
kl_weight=1e-4
torsionloss_weight = 0.5

from typing import List
from torch import Tensor

class RMSD(torch.nn.Module):
    def __init__(self) -> None:
        super(RMSD, self).__init__()

    def forward(self, protein_coords_pred: List[Tensor], protein_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for protein_coords_pred, protein_coords in zip(protein_coords_pred, protein_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((protein_coords_pred - protein_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()
    
def eval(model, loader, length=None):
    """
    Evaluate the model
    Args:
        model: model to evaluate
        data: valid_loader or test_loader
    Returns:
        Model error on data
    """
    model.eval()
    
    total_kl_x = 0
    total_kl_h = 0

    pred_torsion = []
    pred_dist = []
    pred_coord = []
    pred_aa_type = []
    pred_pad_type = []

    true_torsion = []
    true_dist = []
    true_coord = []
    true_aa_type = []
    true_pad_type = []

    count = 0

    rmsd_criterion = RMSD()

    # loop over minibatches
    for step, batch in enumerate(loader):
        count += 1

        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)

        with torch.no_grad():
            pred_coords_ca, pred_residue, pred_pad, kl_x, kl_h = model(batch)


        total_kl_x += kl_x
        total_kl_h += kl_h

        pred_coords_ca_split = torch.split(pred_coords_ca, length)
        protein_mask_split = torch.split(batch.protein_mask, length)
        for pred_coords_ca, protein_mask in zip(pred_coords_ca_split, protein_mask_split):
            pred_coord.append(pred_coords_ca[protein_mask].detach().cpu())
            pred_dist.append((pred_coords_ca[protein_mask][0:-1] - pred_coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt().detach().cpu())
            pred_torsion.append(torsion_angle(pred_coords_ca[protein_mask]))

        pred_aa_type.append(pred_residue[batch.protein_mask].detach().cpu())
        pred_pad_type.append(pred_pad.detach().cpu())

        coords_ca_split = torch.split(batch.coords_ca, length)
        for coords_ca, protein_mask in zip(coords_ca_split, protein_mask_split):
            true_coord.append(coords_ca[protein_mask].detach().cpu())
            true_dist.append((coords_ca[protein_mask][0:-1] - coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt().detach().cpu())
            true_torsion.append(torsion_angle(coords_ca[protein_mask]))

        true_aa_type.append(batch.x[batch.protein_mask].detach().cpu())
        true_pad_type.append(batch.padding.detach().cpu())

    # accuracy for residue type prediction
    preds = torch.argmax(torch.cat(pred_aa_type, dim=0), dim=1)
    acc_residue = torch.sum(preds == torch.cat(true_aa_type, dim=0).squeeze(1)) / preds.shape[0]

    # accuracy for padding type prediction
    preds = (torch.cat(pred_pad_type, dim=0) > 0.5).to(torch.int)
    acc_padding = torch.sum(preds.squeeze(1) == torch.cat(true_pad_type, dim=0)) / preds.shape[0]

    # MAE for atom position reconstruction
    mae = reg_criterion(torch.cat(pred_coord, dim=0), torch.cat(true_coord, dim=0))

    # calculate rmsd, note: this calculation doesn't use alignment, if use krmsd, batch size need to be set to 1
    rmsd = rmsd_criterion(pred_coord, true_coord)

    # MAE for edge distance
    edge_mae = reg_criterion(torch.cat(pred_dist, dim=0), torch.cat(true_dist, dim=0))
    pred_dist = torch.cat(pred_dist, dim=0)
    stable = np.logical_and((pred_dist.cpu().numpy() > 3.65), (pred_dist.cpu().numpy() < 3.95))
    edge_stable = stable.sum() / stable.size

    # MAE for torsion angle
    torsion_mae = reg_criterion(torch.cat(pred_torsion, dim=0), torch.cat(true_torsion, dim=0))

    return torsion_mae, edge_mae, edge_stable, mae, rmsd, acc_residue, acc_padding, total_kl_x / (step + 1), total_kl_h / (step + 1), pred_coord, true_coord, pred_aa_type, true_aa_type, pred_pad_type, true_pad_type

# train
best_valid_rmsd = 1000
best_res_acc = 0
best_edge_stable = 0
best_valid_rmsd_epoch = 0
best_res_acc_epoch = 0
best_edge_stable_epoch = 0
log_file = open('log.txt', 'a')

for epoch in range(1, 3): # 200
    
    total_loss = 0
    total_mae = 0
    total_res_loss = 0
    total_pad_loss = 0
    total_edge_mae = 0
    total_torsion_mae = 0
    total_kl_x = 0
    total_kl_h = 0
    
    log_file = open('log.txt', 'a')
    print(file=log_file)
    print('Epoch', epoch, file=log_file)

    start = time.time()
    model.train()
    for step, batch in enumerate(train_loader):
        print('step', step, file=log_file)
        if step == 2:
            break
        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)

        pred_coords_ca, pred_residue, pred_pad, kl_x, kl_h = model(batch) # model prediction

        assert torch.isnan(pred_coords_ca).sum() == 0
        assert torch.isnan(pred_residue).sum() == 0
        assert torch.isnan(pred_pad).sum() == 0

        # MAE loss
        loss_coords_ca = reg_criterion(pred_coords_ca[batch.protein_mask], batch.coords_ca[batch.protein_mask])
        # cross entropy loss for residue type prediction
        loss_multi_classify = multi_class_criterion(pred_residue[batch.protein_mask], batch.x[batch.protein_mask].squeeze(1).to(torch.long))
        # cross entropy loss for padding type prediction (binary type)
        loss_binary_classify = binary_class_criterion(pred_pad.float(), batch.padding.unsqueeze(1).to(torch.float))

        # edge distance loss
        edge_dist_loss = 0
        pred_coords_ca_split = torch.split(pred_coords_ca, length)
        coords_ca_split = torch.split(batch.coords_ca, length)
        protein_mask_split = torch.split(batch.protein_mask, length)
        count = 0
        for pred_coords_ca, coords_ca, protein_mask in zip(pred_coords_ca_split, coords_ca_split, protein_mask_split):
            count += 1
            pred_dist = (pred_coords_ca[protein_mask][0:-1] - pred_coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt()
            true_dist = (coords_ca[protein_mask][0:-1] - coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt()
            edge_dist_loss += reg_criterion(pred_dist, true_dist)
        edge_dist_loss = edge_dist_loss / count

        # torsion angle loss
        torsion_loss = 0

        for pred_coords_ca, coords_ca, protein_mask in zip(pred_coords_ca_split, coords_ca_split, protein_mask_split):
            count += 1

            pred_torsion = torsion_angle(pred_coords_ca[protein_mask])
            true_torsion = torsion_angle(coords_ca[protein_mask])

            torsion_loss += reg_criterion(pred_torsion, true_torsion)
        torsion_loss = torsion_loss / count

        loss = loss_coords_ca + loss_multi_classify + loss_binary_classify + 0.1 * kl_x + kl_weight * kl_h + edgeloss_weight * edge_dist_loss + torsionloss_weight * torsion_loss

        # reset accumlated gradient from previous backprop and back prop
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu()
        loss_coords_ca = loss_coords_ca.detach().cpu()
        loss_binary_classify = loss_binary_classify.detach().cpu()
        loss_multi_classify = loss_multi_classify.detach().cpu()
        edge_dist_loss = edge_dist_loss.detach().cpu()
        torsion_loss = torsion_loss.detach().cpu()
        if type(kl_x) != int:
            kl_x = kl_x.detach().cpu()
        if type(kl_h) != int:
            kl_h = kl_h.detach().cpu()
        
        # log
        print('loss:', loss, file=log_file)
        print('loss_coords_ca:', loss_coords_ca, file=log_file)
        print('loss_binary_classify:', loss_binary_classify, file=log_file)
        print('loss_multi_classify:', loss_multi_classify, file=log_file)
        print('edge_dist_loss:', edge_dist_loss, file=log_file)
        print('torsion_loss:', torsion_loss, file=log_file)
        print('kl_x:', kl_x, file=log_file)

        total_loss += loss
        total_mae += loss_coords_ca
        total_pad_loss += loss_binary_classify
        total_res_loss += loss_multi_classify
        total_edge_mae += edge_dist_loss
        total_torsion_mae += torsion_loss
        if type(kl_x) != int:
            total_kl_x += kl_x
        if type(kl_h) != int:
            total_kl_h += kl_h

    training_end = time.time()
    
    # Validation
    valid_torsion_mae, valid_edge_mae, edge_stable, valid_mae, rmsd, res_acc, pad_acc, kl_x, kl_h, pred_coord, true_coord, pred_aa_type, true_aa_type, pred_pad_type, true_pad_type = eval(model, valid_loader, length)
    validation_end = time.time()
    

    print('Epoch Train Summary', file=log_file)
    print('training took', (training_end - start)/60, 'seconds', file=log_file)
    
    print('mean_loss:', total_loss / (step + 1), file=log_file)
    print('mean_mae:', total_mae / (step + 1), file=log_file)
    print('mean_res_loss:', total_res_loss / (step + 1), file=log_file)
    print('mean_edge_mae:', total_edge_mae / (step + 1), file=log_file)
    print('mean_torsion_mae:', total_torsion_mae / (step + 1), file=log_file)
    print('mean_kl_x:', total_kl_x / (step + 1), file=log_file)
    print('mean_kl_h:', total_kl_h / (step + 1), file=log_file)

    print('Epoch Validation Summary', file=log_file)
    print('validation took', (validation_end - training_end)/60, 'seconds', file=log_file)
    print('valid_torsion_mae', valid_torsion_mae, file=log_file)
    print('valid_edge_mae', valid_edge_mae, file=log_file)
    print('valid_edge_stable', edge_stable, file=log_file)
    print('valid_mae', valid_mae, file=log_file)
    print('valid_acc_res', res_acc, file=log_file)
    print('valid_acc_pad', pad_acc, file=log_file)
    print('valid_rmsd', rmsd, file=log_file)
    print('valid_kl_x', kl_x, file=log_file)
    print('valid_kl_h', kl_h, file=log_file)

    log_file.close()

    # save model
    if rmsd < best_valid_rmsd:
        best_valid_rmsd = rmsd
        best_valid_rmsd_epoch = epoch
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                        'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
        torch.save(checkpoint, 'checkpoint_bst_rmsd.pt')

    if res_acc > best_res_acc:
        best_res_acc = res_acc
        best_res_acc_epoch = epoch
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                        'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
        torch.save(checkpoint, 'checkpoint_bst_rec_acc.pt')

    if edge_stable > best_edge_stable:
        best_edge_stable = edge_stable
        best_edge_stable_epoch = epoch
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                        'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
        torch.save(checkpoint, 'checkpoint_bst_edge_stable.pt')

log_file = open('log.txt', 'a')
print('\nCheckpoint Summary', file=log_file)
print('best rmsd at epoch', best_valid_rmsd_epoch, file=log_file)
print('best res acc at epoch', best_res_acc_epoch, file=log_file)
print('best edge stable at epoch', best_edge_stable_epoch, file=log_file)
log_file.close()
