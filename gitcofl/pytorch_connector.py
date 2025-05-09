import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary as summary_info
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import glob
import json
import copy

## constants
global_model_pattern = '*-global_weight.pth'
decen_model_pattern = '*-decen_merged_weight.pth'

# load_model represents the function that loads the model from the previous round, it wikk return the model and a boolean value that indicates if the model was loaded successfully
def load_model(model, device, count_fl_round, model_path, fl_type):
    net = model.to(device)
    ## case first round of clients, load empy model
    if count_fl_round == 1:
        print("for the first round, using model is empty")
        return net, True
    ## case load global weights (another round)
    else:
        if os.path.exists(model_path) and os.path.isdir(model_path):
            if fl_type == "cen":
                matching_files = glob.glob(os.path.join(model_path, global_model_pattern))
            else:
                matching_files = glob.glob(os.path.join(model_path, decen_model_pattern))

            if matching_files:
                file_to_open = matching_files[0]
                net.load_state_dict(torch.load(file_to_open))

                # Todo: check about input size, should it dynamically config?
                print(summary_info(net, input_size = (32, 3, 32, 32))) # (batchsize,channel,width,height)
                net = net.to(device)
                print(f"model loaded from {file_to_open}")

                # Remove all matching files
                if fl_type == "decen":
                    for f in matching_files:
                        try:
                            os.remove(f)
                            print(f"Deleted temp aggregated model file: {f}")
                        except Exception as e:
                            print(f"Failed to delete {f}: {e}")
                    
                return net, True
            
        return None, False

# train represents the function that trains the model on the local client
def train(net, trainloader, valloader, local_epochs, device, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

    history_train = {'loss':np.zeros(local_epochs), 'acc':np.zeros(local_epochs), 'f1-score':np.zeros(local_epochs)}
    history_val = {'loss':np.zeros(local_epochs), 'acc':np.zeros(local_epochs), 'f1-score':np.zeros(local_epochs)}
    min_val_loss = 1e10
    patience_counter = 0

    for epoch in range(local_epochs):
        print(f'epoch {epoch + 1} \nTraining ...')
        y_predict = list()
        y_labels = list()
        training_loss = 0.0
        n = 0
        net.train()
        for data in tqdm(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            n+=1

            y_labels += list(labels.cpu().numpy())
            y_predict += list(outputs.argmax(dim=1).cpu().numpy())

        # print statistics
        report = classification_report(y_labels, y_predict, digits=4, output_dict=True, zero_division=0)
        acc = report["accuracy"]
        f1 = report["weighted avg"]["f1-score"]
        support = report["weighted avg"]["support"]
        training_loss /= n
        print(f"training loss: {training_loss:.4}, acc: {acc*100:.4}%, f1-score: {f1*100:.4}%, support: {support}" )
        history_train['loss'][epoch] = training_loss
        history_train['acc'][epoch] = acc
        history_train['f1-score'][epoch] = f1

        print('validating ...')
        net.eval()
        y_predict = list()
        y_labels = list()
        validation_loss = 0.0
        n = 0
        with torch.no_grad():
            for data in tqdm(valloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                y_labels += list(labels.cpu().numpy())
                y_predict += list(outputs.argmax(dim=1).cpu().numpy())
                n+=1

        # print statistics
        report = classification_report(y_labels, y_predict, digits = 4, output_dict = True)
        acc = report["accuracy"]
        f1 = report["weighted avg"]["f1-score"]
        support = report["weighted avg"]["support"]
        validation_loss /= n
        print(f"validation loss: {validation_loss:.4}, acc: {acc*100:.4}%, f1-score: {f1*100:.4}%, support: {support}" )
        history_val['loss'][epoch] = validation_loss
        history_val['acc'][epoch] = acc
        history_val['f1-score'][epoch] = f1

        #save min validation loss
        if validation_loss < min_val_loss:
            # torch.save(net.state_dict(), MODE_PATH)
            min_val_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epoch(s)')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print('Finished Training')

    return net, criterion

# test represents the function that tests the model on the local client
def test(net, testloader, criterion, device):
    print('testing ...')
    y_predict = list()
    y_labels = list()
    test_loss = 0.0
    n = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            y_labels += list(labels.cpu().numpy())
            y_predict += list(outputs.argmax(dim=1).cpu().numpy())
            n += 1

        # Compute test loss
        test_loss /= n
        print(f"testing loss: {test_loss:.4}" )

        # Compute accuracy
        accuracy = accuracy_score(y_labels, y_predict) * 100
        print(f"Accuracy: {accuracy:.2f}%")

        # Print classification report
        report = classification_report(y_labels, y_predict, digits=4)
        print(report)

        return test_loss, accuracy, len(testloader.dataset)
    
def save_model(net ,model_path, client_id):
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    torch.save(net.state_dict(), os.path.join(model_path, f'{timestamp}-weight-client-{client_id}.pth'))

def agg_fed_avg(local_models: list, model, model_path: str, fl_type: str):
    """
    Perform FedAvg aggregation on the local models.
    """
    global_model = copy.deepcopy(model)

    for weight_file in local_models:
        client_model = model
        client_model.load_state_dict(torch.load(os.path.join(model_path, weight_file)))

        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            global_param.data += client_param.data

    num_clients = len(local_models)
    for global_param in global_model.parameters():
        global_param.data /= num_clients

    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    if fl_type == "cen":
        new_weight_file = f'{timestamp}-global_weight.pth'
    else:
        new_weight_file = f'{timestamp}-decen_merged_weight.pth'

    torch.save(global_model.state_dict(), os.path.join(model_path, new_weight_file))

    return new_weight_file
