import torch.nn as nn

def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class SyntheticGaussianNet_Enc(nn.Module):
    def __init__(self,  args, encs=False):
        super(SyntheticGaussianNet_Enc, self).__init__()

        self.args = args
        input_size1 = args.get("input_size", 100)
        hidden_size = args.get("hidden_size", 100)

        # print(self.args)
        layers=2
        activation='relu'

        if self.args.modality == 0:
            layers = self.args.get("layers", 2)
            if layers == 5:
                hidden_size = 500
            elif layers == 3:
                hidden_size = 200

        self.enc = mlp(input_size1, hidden_size, hidden_size, layers, activation)
        self.projection = mlp(hidden_size, hidden_size, 100, 1, activation)
        self.pred_fc = nn.Linear(100, args.num_classes)

    def forward(self, x, **kwargs):

        x1 = x[self.args.modality]
        out1 = self.projection(self.enc(x1))
        pred = self.pred_fc(out1)

        return {"preds":{"combined":pred}, "features": {"combined": out1}}

