import torch.nn as nn
import torch
from collections import OrderedDict

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
        x =self.dropout(x)
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

    def summary(self, input_size):
       	def register_hook(module):
            def hook(module, input, output):
                if module._modules:  # only want base layers
                    return
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = None
                if output.__class__.__name__ == 'tuple':
                    summary[m_key]['output_shape'] = list(output[0].size())
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = None

                params = 0
                # iterate through parameters and count num params
                for name, p in module._parameters.items():
                    params += torch.numel(p.data)
                    summary[m_key]['trainable'] = p.requires_grad

                summary[m_key]['nb_params'] = params

            if not isinstance(module, torch.nn.Sequential) and \
               not isinstance(module, torch.nn.ModuleList) and \
               not (module == self):
                hooks.append(module.register_forward_hook(hook))

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [(torch.rand(1, *in_size)) for in_size in input_size]
        else:
            x = (torch.randn(1, *input_size))

        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        self.apply(register_hook)
        # make a forward pass
        self(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        # print out neatly
        def get_names(module, name, acc):
            if not module._modules:
                acc.append(name)
            else:
                for key in module._modules.keys():
                    p_name = key if name == "" else name + "." + key
                    get_names(module._modules[key], p_name, acc)
        names = []
        get_names(self, "", names)

        col_width = 25  # should be >= 12
        summary_width = 61

        def crop(s):
            return s[:col_width] if len(s) > col_width else s

        print('_' * summary_width)
        print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
            'Layer (type)', 'Output Shape', 'Param #', col_width))
        print('=' * summary_width)
        total_params = 0
        trainable_params = 0
        for (i, l_type), l_name in zip(enumerate(summary), names):
            d = summary[l_type]
            total_params += d['nb_params']
            if 'trainable' in d and d['trainable']:
                trainable_params += d['nb_params']
            print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
                crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
                crop(str(d['nb_params'])), col_width))
            if i < len(summary) - 1:
                print('_' * summary_width)
        print('=' * summary_width)
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str((total_params - trainable_params)))
        print('_' * summary_width)
