import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def _init_(self):
        super(CNN, self)._init_()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 2),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Dropout(0.8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.Sigmoid(),
            nn.Dropout(0.7),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(16 * 7 * 7, 10)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return F.log_softmax(output, dim=1)


class Digit_Classifier(nn.Module):
    def _init_(self):
        super(Digit_Classifier, self)._init_()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.65)
        self.out = nn.Linear(128, 10)

    def forward(self, inputs):
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x


class MNIST_Net(nn.Module):
    def _init_(self):
        super(MNIST_Net, self)._init_()
        self.layer1 = nn.Conv2d(1, 10, kernel_size=5)
        self.layer2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(0.6)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.layer1(x), 2))
        x = self.dropout(F.relu(F.max_pool2d(self.layer2(x), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Model 3: Sequential CNN with heavy dropout


# Class for running and evaluating models
class RunModel():
    def _init_(self):
        self.loaded = False

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    def train(self, model, epochs=5, learning_rate=0.01):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(self.trainloader)
            epoch_losses.append(avg_loss)
            print(f'[Model {model._class.name_}] Epoch {epoch + 1} Average Loss: {avg_loss:.3f}')
        print(f'Finished Training {model._class.name_}')
        return model, epoch_losses

    def test_one_image(self, model):
        model.eval()
        data_iter = iter(self.testloader)
        images, labels = next(data_iter)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print(f'[Model {model._class.name_}] Predicted Label: {predicted.item()}, Actual Label: {labels.item()}')

    def test(self, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'[Model {model._class.name_}] Accuracy on Test Set: {accuracy:.2f}%')
        return accuracy

def plot_epoch_losses(models_losses):
    for model_name, losses in models_losses.items():
        plt.plot(losses, label=f'{model_name}')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
run_model = RunModel()
run_model.load()

models = [ CNN(), Digit_Classifier(), MNIST_Net()]
models_losses = {}
models_accuracies = {}
for model in models:
    trained_model, losses = run_model.train(model)
    accuracy = run_model.test(trained_model)
    models_losses[model._class.name_] = losses
    models_accuracies[model._class.name_] = accuracy
    run_model.test_one_image(trained_model)

plot_epoch_losses(models_losses)

# Print model accuracies
for model_name, accuracy in models_accuracies.items():
    print(f'Model {model_name}: Accuracy on Test Set: {accuracy:.2f}%')