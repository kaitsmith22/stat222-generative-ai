import torch, os
import time
import pandas as pd

from torch.optim import lr_scheduler

class VAEExperiment():
    """
    Class to handle training and saving of model outputs
    """

    def __init__(self, model, datasets, learning_rate, gamma, device, name, base_dir, num_epoch):
        """

        :param model: model to train
        :param datasets: dictionary with 2 Pytorch dataloaders: one with key 'train' and one with key 'test'
        :param learning_rate: learning rate of optimizer
        :param gamma: parameter for exponential learning rate scheduler
        :param device: device to perform training on
        :param name: name of experiment. a folder will be created with this name to save outputs
        :param base_dir: name of base directory to save outputs
        :param num_epoch: number of epochs to train
        """

        self.model = model

        self.name = name

        self.num_epochs = num_epoch

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.dataloaders = datasets

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,
                                             gamma = gamma)

        self.device = device

        # change to pandas dataframe
        self.train_loss = []
        self.val_loss = []

        self.dir = os.path.join(base_dir, name)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)



    def train_model(self):
        """
            Training loop for the experiment
        """
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)


            for phase in ['train', 'val']:
                if phase == 'train':
                    # inform layers such as BatchNorm that we are training
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                # iterate through each training and validation dataloader
                for image, mask in self.dataloaders[phase]:

                    # put image and mask on the device
                    image = image.to(self.device)
                    mask = mask.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(mask)
                        total_loss = self.model.loss_function(outputs[0], image, outputs[1], outputs[2])
                        # print('all losses: ', total_loss)
                        loss = total_loss['loss']

                        # propagate loss and step optimizer if in training stage
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # multiply by batch size (image.size(0)), because we want to find loss for the entire epoch,
                    # and not just for loss of the batch
                    running_loss += loss * image.size(0)
                    break

                # update the learning rate if this is training iteration
                if phase == 'train':
                    self.scheduler.step()

                # epoch loss is the total loss / the length of the dataset
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)

                if phase == "train":
                    self.train_loss.append(epoch_loss.detach().cpu().squeeze()) # put loss on cpu and remove extra dimention

                else:
                    self.val_loss.append(epoch_loss.detach().cpu().squeeze())

                print(f'{phase} Loss: {epoch_loss:.4f}')


        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    def save_model(self):
        """
        Function to save the model weights at [base_dir]/[name]/name_weights.pt
        :return: None
        """

        path = os.path.join(self.dir, self.name + "_weights.pt")
        torch.save(self.model.state_dict(), path)
        print("Saved model weights at ", path)

    def load_model(self,path):
        """
        Function to load the model weights at [base_dir]/[name]/name_weights.pt
        :return: None
        """
        self.model.load_state_dict(path)

    def plot_training(self):
        """
        TODO: IMPLEMENT!!!
        :param self:
        :type self:
        :return:
        :rtype:
        """
        path = os.path.join(self.dir, self.name + "_training_metrics.png")

    def save_metrics(self):
        """
        Save training metrics to a csv file in the directory for the experiment

        :return:None
        """
        df = pd.DataFrame({'Train_Loss': self.train_loss, 'Val_Loss': self.val_loss})
        path = os.path.join(self.dir, self.name + "_training_metrics.csv")
        df.to_csv(path)
        print('Saved Training Metrics at '+ path)




