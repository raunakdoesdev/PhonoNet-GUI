import torch.nn as nn
import torch
from collections import OrderedDict
from matplotlib.cm import get_cmap


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_dim),
                torch.zeros(self.num_layers, 1, self.hidden_dim))

    def forward(self, x):
        x = self.dropout(x)
        lstm_out, self.hidden = self.lstm(x)
        return lstm_out


class RagaDetector(nn.Module):
    def __init__(self, dropout=0.15, hidden_size=256):
        super(RagaDetector, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)),

            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),

            ('conv3', nn.Conv2d(128, 150, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(150)),
            ('relu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool2d([4, 2])),
            ('drop3', nn.Dropout(p=dropout)),

            ('conv4', nn.Conv2d(150, 200, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(200)),
            ('gba', nn.AvgPool2d([3, 125])),
            ('drop4', nn.Dropout(p=dropout))
        ]))

        self.fc1 = nn.Linear(200, 30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = self.fc1(x)
        return x


class SalientRagaDetector(nn.Module):
    def __init__(self, raga_detector):  # Create new broken down raga detector based on existing model
        super(SalientRagaDetector, self).__init__()
        self.encoder1 = nn.Sequential(raga_detector.encoder[0:6])
        self.encoder2 = nn.Sequential(raga_detector.encoder[6:11])
        self.encoder3 = nn.Sequential(raga_detector.encoder[11:16])
        self.encoder4 = nn.Sequential(raga_detector.encoder[16:20])
        self.fc1 = raga_detector.fc1

    def forward(self, x):
        out0 = x.mean(1).unsqueeze(0)

        self.up1 = nn.Upsample([x.shape[2], x.shape[3]], mode='nearest')
        x = self.encoder1(x)
        out1 = x.mean(1).unsqueeze(0)

        self.up2 = nn.Upsample([x.shape[2], x.shape[3]], mode='nearest')
        x = self.encoder2(x)
        out2 = x.mean(1).unsqueeze(0)

        self.up3 = nn.Upsample([x.shape[2], x.shape[3]], mode='nearest')
        x = self.encoder3(x)
        out3 = x.mean(1).unsqueeze(0)

        self.up4 = nn.Upsample([x.shape[2], x.shape[3]], mode='nearest')
        x = self.encoder4(x)
        out4 = x.mean(1).unsqueeze(0)

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Magical line that scales everything and does the mathsss
        activation_map = torch.mul(self.up1(torch.mul(self.up2(torch.mul(self.up3(out3), out2)), out1)), out0)

        return x, activation_map
