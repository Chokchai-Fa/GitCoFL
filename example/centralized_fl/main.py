from gitcofl.cen_fl import CentralizedFL
from dotenv import load_dotenv
import os
from datetime import datetime

from model import CNN, load_data_set
from sklearn.metrics import classification_report
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


FL_INTERVAL = 10 
REPO_PATH = "./fl_process"
load_dotenv()

# Initialize centralized federated learning with a Git repository and scheduling
fl = CentralizedFL(
    repo_path=REPO_PATH, 
    git_repo_url=os.environ.get('GIT_FL_REPO'), 
    access_token=os.environ.get('GIT_ACCESS_TOKEN'), 
    interval=FL_INTERVAL)

fl.start()  # Starts the federated learning process with a scheduled check every 30 seconds




MODE_PATH = './CNN_CIFAR10.pth'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)


trainloader, testloader, valloader = load_data_set('./data')




def train_model(net):
    # global directory_path
    # directory_path = f"{repo_path}/round{count_fl_round}"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)


    epochs = 5

    history_train = {'loss':np.zeros(epochs), 'acc':np.zeros(epochs), 'f1-score':np.zeros(epochs)}
    history_val = {'loss':np.zeros(epochs), 'acc':np.zeros(epochs), 'f1-score':np.zeros(epochs)}
    min_val_loss = 1e10
    # PATH = './'+client_id+'/CNN_CIFAR10.pth'

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'epoch {epoch + 1} \nTraining ...')
        y_predict = list()
        y_labels = list()
        training_loss = 0.0
        n = 0
        net.train()
        for data in tqdm(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # forward
            loss = criterion(outputs, labels) # calculate loss from forward pass
            loss.backward() # just calculate
            optimizer.step() # update weights here

            # aggregate statistics
            training_loss += loss.item()
            n+=1

            y_labels += list(labels.cpu().numpy())
            y_predict += list(outputs.argmax(dim=1).cpu().numpy())

        # print statistics
        report = classification_report(y_labels, y_predict, digits = 4, output_dict = True)
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
            # files = os.listdir(directory_path)
            # for file_name in files:
            #     file_path = os.path.join(directory_path, file_name)
            #     if os.path.isfile(file_path):
            #         os.remove(file_path)

            # current_datetime = datetime.now()
            # timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            # torch.save(net.state_dict(), os.path.join(directory_path, f'{timestamp}-weight-client-{client_id}.pth'))
            # min_val_loss = validation_loss
            torch.save(net.state_dict(), MODE_PATH)
            min_val_loss = validation_loss

    print('Finished Training')

    # net = CNN().to(device)

    # directory_path = f"{repo_path}/round{count_fl_round}"
    # if os.path.exists(directory_path) and os.path.isdir(directory_path):
    #     matching_files = glob.glob(os.path.join(directory_path, global_model_pattern))
    #     if matching_files:
    #         file_to_open = matching_files[0]

    #         print("hello0", file_to_open)
    #         net.load_state_dict(torch.load(file_to_open))
    
    return net, criterion


def test_model(net, criterion):
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
            n+=1

        # print statistics
        test_loss /= n
        print(f"testing loss: {test_loss:.4}" )

        report = classification_report(y_labels, y_predict, digits = 4)
        print(report)



