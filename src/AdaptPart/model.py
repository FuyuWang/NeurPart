import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout
    ):
        super(TemporalBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        self.ll_conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)

        self.ll_conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.sigmoid = nn.Sigmoid()

    def net(self, x, block_num, params=None):
        layer_name = "ll_tc.ll_temporal_block" + str(block_num)
        if params is None:
            x = self.ll_conv1(x)
        else:
            x = F.conv1d(
                x,
                weight=params[layer_name + ".ll_conv1.weight"],
                bias=params[layer_name + ".ll_conv1.bias"],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        x = self.chomp1(x)
        x = F.leaky_relu(x)

        return x

    def init_weights(self):
        self.ll_conv1.weight.data.normal_(0, 0.01)
        self.ll_conv2.weight.data.normal_(0, 0.01)

    def forward(self, x, block_num, params=None):
        out = self.net(x, block_num, params)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            setattr(
                self,
                "ll_temporal_block{}".format(i),
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                ),
            )

    def forward(self, x, params=None):

        for i in range(self.num_levels):
            temporal_block = getattr(self, "ll_temporal_block{}".format(i))
            x = temporal_block(x, i, params=params)
        return x

def init_weights(m):
    if type(m) == nn.LSTMCell:
        torch.nn.init.orthogonal_(m.weight_hh)
        torch.nn.init.orthogonal_(m.weight_ih)

class Actor(nn.Module):
    def __init__(self, dim_size, resource_size, n_action_steps, action_size, h_size=128, hidden_dim=10, num_steps=10):
        super(Actor, self).__init__()
        self.encoder = nn.Linear(dim_size, hidden_dim)
        self.encoder_status = nn.Linear(1, hidden_dim)
        self.encoder_action = torch.nn.Linear(n_action_steps, hidden_dim)
        self.fc10 = nn.Linear(resource_size, hidden_dim)
        self.fc11 = nn.Linear(3*hidden_dim, h_size)
        self.fc12 = nn.Linear(h_size, h_size)
        self.fc13 = nn.Linear(h_size, h_size)

        self.decoder1 = nn.Linear(h_size, h_size)
        self.decoder2 = nn.Linear(h_size, h_size)
        self.output1 = nn.Linear(h_size, action_size[0])
        self.output2 = nn.Linear(h_size, action_size[1])
        self.lstm = torch.nn.LSTMCell(h_size, h_size)
        self.n_action_steps = n_action_steps

        self.learned_input = None
        self.num_steps = num_steps
        self.ll_tc = TemporalConvNet(
            self.num_steps, [10, 1], kernel_size=2, dropout=0.0
        )

        self.init_weight()
    def init_weight(self):
        self.apply(init_weights)
    def forward(self, dimension, action_status, action_val, action_step, lstm_hid,temperature=1, params=None, pe_constraints={}):
        if params is None:
            x1 = self.encoder(dimension)
            x1 = x1.unsqueeze(0)
            x2 = self.encoder_action(action_val)
            x2 = x2.unsqueeze(0)
            x3 = self.encoder_status(action_status)
            x3 = x3.unsqueeze(0)
            x = torch.cat((x1, x2,x3), dim=1)
            x = F.relu(self.fc11(x))
            x = F.relu(self.fc12(x))
            x = F.relu(self.fc13(x))
            hx, cx = self.lstm(x, lstm_hid)

            x = hx
            x1 = F.relu(self.decoder1(x))
            x1 = self.output1(x1)

            x2 = F.relu(self.decoder2(x))
            x2 = self.output2(x2)
        else:
            x1 = F.linear(
                    dimension,
                    weight=params["encoder.weight"],
                    bias=params["encoder.bias"],
                )
            x1 = x1.unsqueeze(0)

            x2 = F.linear(
                    action_val,
                    weight=params["encoder_action.weight"],
                    bias=params["encoder_action.bias"],
                )
            x2 = x2.unsqueeze(0)

            x3 = F.linear(
                    action_status,
                    weight=params["encoder_status.weight"],
                    bias=params["encoder_status.bias"],
                )
            x3 = x3.unsqueeze(0)

            x = torch.cat((x1, x2,x3), dim=1)
            x = F.relu(F.linear(
                    x,
                    weight=params["fc11.weight"],
                    bias=params["fc11.bias"],
                    )
                )
            x = F.relu(F.linear(
                    x,
                    weight=params["fc12.weight"],
                    bias=params["fc12.bias"],
                    )
                )
            x = F.relu(F.linear(
                    x,
                    weight=params["fc13.weight"],
                    bias=params["fc13.bias"],
                    )
                )
            hx, cx = torch._VF.lstm_cell(
                x,
                lstm_hid,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
                )

            x = hx
            x1 = F.relu(F.linear(
                    x,
                    weight=params["decoder1.weight"],
                    bias=params["decoder1.bias"],
                    )
                )
            x1 = F.linear(
                    x1,
                    weight=params["output1.weight"],
                    bias=params["output1.bias"],
                )

            x2 = F.relu(F.linear(
                    x,
                    weight=params["decoder2.weight"],
                    bias=params["decoder2.bias"],
                    )
                )
            x2 = F.linear(
                    x2,
                    weight=params["output2.weight"],
                    bias=params["output2.bias"],
                )
        remained_pes = pe_constraints['remained_pes']
        pe_space = pe_constraints['action_space'][0]
        ktile_space = pe_constraints['action_space'][1]
        ktile_size_max = pe_constraints['ktile_size_max']

        pe_mask = torch.from_numpy(pe_space > remained_pes).to(dimension.device).float() * -1000000.
        x1 = F.softmax(x1/temperature + pe_mask, dim=1)

        ktile_mask = torch.from_numpy(ktile_space > ktile_size_max).to(x1.device).float() * -1000000.
        x2 = F.softmax(x2 / temperature + ktile_mask, dim=1)
        return x1, x2, (hx, cx)

    def learned_loss(self, H, params=None):
        if H.size(0) < 6:
            return None
        H = H[-self.num_steps:]
        H_input = H.unsqueeze(0)
        x = self.ll_tc(H_input, params).squeeze()
        return x.pow(2).sum(0).pow(0.5)

