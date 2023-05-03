import time
from tqdm import tqdm
import copy

import numpy as np

# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from adabelief_pytorch import AdaBelief

# from global_constants import left_handed_patients
from models import LSTM, BERTModelCostum, BERTClassifierCostum, CNN1D_LSTM, CNN1D
from focal_loss import FocalLoss
from ETL import sample_based_data_transform

# region_code_np = np.asarray(region_code)
# label_test = [r[1] for r in region_map.items()]
# self.text_labels = region_code_np[np.sort(np.unique(label_test)).astype(int)].tolist()
# cfm = self.get_cfm()
# df_cm = pd.DataFrame(
#     cfm, index=[i for i in self.text_labels], columns=[i for i in self.text_labels]
# )
# plt.ioff()
# figure = plt.figure(figsize=(20, 10))
# sn.heatmap(df_cm, annot=True)
# plt.savefig(fn + ".jpg")
# plt.close(figure)

def initialize_training(training_config, eval_config, segment_config, patient_id=None):
    # self.model = MLP(output_size=16).to(self.computing_device)

    # is_left_handed = patient_id in left_handed_patients
    if training_config['model_type'].lower() == 'xgb':
        model = xgb.XGBClassifier(n_estimators=training_config['model_config']['XGB_config']['num_trees'], max_depth=training_config['model_config']['XGB_config']['trees_depth'])
        return model
    elif training_config['model_type'].lower() == 'rf':
        model = RandomForestClassifier(n_estimators=training_config['model_config']['XGB_config']['num_trees'], max_depth=training_config['model_config']['XGB_config']['trees_depth'])
        return model
    elif training_config['model_type'].lower() == 'bert':
        # model = BERTModelCostum(training_config['model_config']['transformer_config'], eval_config, segment_config).to(training_config['computing_device'])
        model = BERTClassifierCostum(training_config['model_config']['transformer_config'], eval_config, segment_config, computing_device=training_config['computing_device']).to(training_config['computing_device'])
        #, is_left_handed
    elif training_config['model_type'].lower() == 'lstm':
        model = LSTM(len(eval_config['data_fields']), eval_config, training_config['model_config']['LSTM_config']).to(training_config['computing_device'])
    elif training_config['model_type'].lower() == 'cnn':
        model = CNN1D(len(eval_config['data_fields']), eval_config['num_dental_regs'], training_config['model_config']['CNN1D_config']).to(training_config['computing_device'])
    elif training_config['model_type'].lower() == 'cnn_lstm':
        model = CNN1D_LSTM(len(eval_config['data_fields']), eval_config['num_dental_regs'], training_config['model_config']['CNN1D_LSTM_config']).to(training_config['computing_device'])

    
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    # optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    
    # if training_config['loss_criterion'].lower() == 'focal_loss':
    #     loss_criterion = FocalLoss(alpha=0.5, gamma=2)
    # else:
    loss_criterion = nn.CrossEntropyLoss().to(training_config['computing_device'])

    return model, optimizer, loss_criterion


def train_model(train_dataloader, model, optimizer, loss_criterion, num_epochs=5, computing_device='cpu', valid_dataloader=None, verbose=True):
    since = time.time()
    model = model.to(computing_device)
    train_loss_all = []
    train_acc_all = []
    best_val_acc = 0
    best_model = None
    val_loss_all = []
    val_acc_all = []
    val_f1_all = []

    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=num_epochs)
    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=np.floor(0.05*num_epochs), cycle_momentum=False, mode="triangular")

    try:
        for _, epoch in tqdm(enumerate(range(num_epochs)), total=num_epochs, desc='Training Progress'):

            model.train()
            train_loss_epoch = 0
            train_accuracy_epoch_unnormalized = 0
            for _, (batch_data, batch_labels, batch_brush_types, batch_is_left_handed) in enumerate(train_dataloader):
                batch_data = batch_data.float().to(computing_device)
                batch_brush_types = batch_brush_types.type(torch.LongTensor).requires_grad_(False).to(computing_device)
                batch_is_left_handed = batch_is_left_handed.type(torch.LongTensor).requires_grad_(False).to(computing_device)
                batch_labels = batch_labels.type(torch.LongTensor).to(computing_device)  #if crossentropyloss
                # batch_labels = batch_labels.reshape(-1, 1).float()

                
                optimizer.zero_grad()
                batch_outputs = model(batch_data, batch_brush_types, batch_is_left_handed)
                batch_loss = loss_criterion(batch_outputs, batch_labels)
                batch_loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss_epoch += batch_loss.item() # we could divide it by the number of samples in the loader but doesnt matter
                # batch_predictions = batch_outputs.detach().cpu().flatten().numpy() > 0.5
                # batch_predictions = torch.sigmoid(batch_outputs).detach().cpu().flatten().numpy() > 0.5
                batch_predictions = np.argmax(batch_outputs.detach().cpu().numpy(), -1)
                batch_predictions = batch_predictions.reshape(-1)
                batch_labels = batch_labels.cpu().numpy().reshape(-1)

                train_accuracy_epoch_unnormalized += \
                    (batch_predictions == batch_labels).sum()/ len(batch_labels)
            
            train_loss_all.append(train_loss_epoch/float(len(train_dataloader)))
            train_acc_all.append(train_accuracy_epoch_unnormalized/float(len(train_dataloader)))
            
            # scheduler.step()

            if verbose:
                print("Epoch %d, loss: %.3f, accuracy: %.3f" % (epoch + 1, \
                    train_loss_all[-1], train_acc_all[-1]))
            
            # Validation
            with torch.no_grad():
                if valid_dataloader is not None:
                    if epoch % 1 == 0:
                        val_loss, val_acc, val_pred, val_gt, val_output = validate_model(valid_dataloader, model, \
                            computing_device=computing_device, loss_criterion=loss_criterion, verbose = verbose)
                        val_loss_all.append(val_loss)
                        val_acc_all.append(val_acc)
                        val_f1_all.append(f1_score(val_gt, val_pred, labels=list(range(0,16)), average='micro'))
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc 
                            best_model = copy.deepcopy(model)

    except KeyboardInterrupt:
        print("Training interrupted after", epoch + 1, "epochs")
        time_elapsed = time.time() - since
        print("Training interrupted in {:.0f}m {:.0f}s ".format(time_elapsed // 60, time_elapsed % 60))
        return best_model, train_loss_all, train_acc_all, val_loss_all, val_acc_all, val_f1_all

    print("Training complete after", epoch + 1, "epochs")
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s ".format(time_elapsed // 60, time_elapsed % 60))
    validate_model(valid_dataloader, best_model, computing_device=computing_device, loss_criterion=loss_criterion, verbose = True)
    # print("--------------testing------------")
    # test_loss, test_acc = validate_model(test_dataloader, model)

    return best_model, train_loss_all, train_acc_all, val_loss_all, val_acc_all, val_f1_all


def validate_model(valid_dataloader, model, computing_device='cpu', loss_criterion=nn.CrossEntropyLoss(), verbose=True):
    start = time.time()
    model.eval()
    valid_loss = 0
    correct_num = 0
    total_num = 0
    valid_outputs = []
    valid_predictions = []
    valid_label = []

    for _, (batch_data, batch_labels, batch_brush_types, batch_is_left_handed) in enumerate(valid_dataloader):
        batch_data = batch_data.float().to(computing_device)
        batch_brush_types = batch_brush_types.type(torch.LongTensor).requires_grad_(False).to(computing_device)
        batch_is_left_handed = batch_is_left_handed.type(torch.LongTensor).requires_grad_(False).to(computing_device)
        batch_labels = batch_labels.type(torch.LongTensor).to(computing_device) #if crossentropyloss
        # batch_labels = batch_labels.reshape(-1, 1).float()
        with torch.no_grad():
            batch_outputs = model(batch_data, batch_brush_types, batch_is_left_handed)
            # batch_outputs = torch.sigmoid(model(batch_data))
            batch_loss = loss_criterion(batch_outputs, batch_labels)
            valid_loss += batch_loss.item()

            valid_outputs.append(batch_outputs.detach().cpu().numpy())
            # batch_predictions = batch_outputs.detach().cpu().flatten().numpy() > 0.5
            batch_predictions = np.argmax(batch_outputs.detach().cpu().numpy(), -1)
            batch_predictions = batch_predictions.reshape(-1)
            batch_labels = batch_labels.cpu().numpy().reshape(-1)
            valid_predictions.append(batch_predictions)
            valid_label.append(batch_labels)

            correct_num += (batch_predictions == batch_labels).sum()
            total_num += len(batch_labels)
        
    valid_loss = valid_loss/float(len(valid_dataloader))
    valid_accuracy = correct_num/total_num
    
    if verbose:
        print("Inference: Time %.3f, loss: %.3f, accuracy: %.3f" % (time.time() - start, valid_loss, valid_accuracy))

    valid_predictions = np.hstack(valid_predictions)
    valid_label = np.hstack(valid_label)
    valid_outputs = np.vstack(valid_outputs)
    
    return valid_loss, valid_accuracy, valid_predictions, valid_label, valid_outputs


def predict_model(test_data, model, computing_device='cpu', loss_criterion=nn.CrossEntropyLoss(), brush_type=0, is_left_handed=0):
    model.eval()

    if type(test_data) == np.ndarray or type(test_data) == list:
        
        brush_type = brush_type.repeat(len(test_data)) #[:,None]
        is_left_handed = is_left_handed.repeat(len(test_data)) #[:,None]

        y_pred = []
        p_pred = []
        for _, data in enumerate(test_data):
            data = torch.tensor(data)
            data = data.float().to(computing_device)
            data = torch.unsqueeze(data, dim=0)
            brush_type = brush_type.type(torch.LongTensor).requires_grad_(False).to(computing_device)
            is_left_handed = is_left_handed.type(torch.LongTensor).requires_grad_(False).to(computing_device)

            with torch.no_grad():
                # batch_outputs = model(data)
                batch_outputs = torch.sigmoid(model(data, brush_type, is_left_handed))

                p_pred.append(batch_outputs.detach().cpu().numpy())
                batch_pred = np.argmax(batch_outputs.detach().cpu().numpy(), -1)
                batch_pred = batch_pred.reshape(-1)
                y_pred.append(batch_pred)

        y_pred = np.hstack(y_pred)
        p_pred = np.vstack(p_pred)

    elif type(test_data) == torch.utils.data.dataloader.DataLoader:  # assume testdata comes as dataloader
        test_loss, test_acc, y_pred, y_label, model_out = validate_model(test_data, model,
                                                                      computing_device=computing_device,
                                                                      loss_criterion=loss_criterion, verbose=False)
        # model_out = torch.tensor(model_out)
        # softmax = nn.Softmax(dim = 1)
        # p_pred = softmax(model_out)[:, 1]
        p_pred = model_out

    else:
        raise NotImplementedError("Use np.ndarray or torch dataloader type as test data!")
    return y_pred, p_pred


def full_train_validate(training_config, eval_config, segment_config, train_loader, valid_loader, test_loader):

    if training_config['model_type'].lower() in ['xgb', 'rf']: 
        model = initialize_training(training_config, eval_config, segment_config)

        train_frs, train_labels = sample_based_data_transform(train_loader)
        valid_frs, valid_labels = sample_based_data_transform(valid_loader)
        test_frs, test_labels = sample_based_data_transform(test_loader)
        
        model.fit(train_frs, train_labels)
        best_model = copy.deepcopy(model)
        valid_predictions = model.predict(valid_frs)
        test_predictions = model.predict(test_frs)
        
        train_accuracy = accuracy_score(train_labels, model.predict(train_frs))
        valid_accuracy = accuracy_score(valid_labels, valid_predictions)
        test_accuracy = accuracy_score(test_labels, test_predictions)
        # test_top2_accuracy = top_k_accuracy_score(test_labels, test_predictions, k=2)

        print(f'\ntest accuracy is {test_accuracy}\n')
        # print(f'\ntest top2 accuracy is {test_top2_accuracy}\n')
    else:
        model, optimizer, loss_criterion = initialize_training(training_config, eval_config, segment_config)
        best_model, _, _, _, _, _ = train_model(train_loader, model, optimizer, loss_criterion, training_config['num_epochs'], \
            training_config['computing_device'], valid_dataloader=valid_loader, verbose=True)
        
        _, test_accuracy, test_predictions, test_labels, _ = validate_model(test_loader, best_model, \
            computing_device=training_config['computing_device'], loss_criterion=loss_criterion, verbose = True)
        
    return best_model, test_accuracy, test_predictions, test_labels

# wandb.log(
#     {
#         "train_loss": loss,
#         "train_acc": acc,
#         "tcfm": cfm_fig,
#         "val_loss": v_loss,
#         "val_accuracy": v_acc,
#         "v_cfm": v_cfm_fig,
#     }
# )
