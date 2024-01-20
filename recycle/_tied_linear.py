import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torch.optim as optim

# tied autoencoder using off the shelf nn modules
class TiedAutoEncoderOffTheShelf(nn.Module):
    def __init__(self, inp, out, weight):
        super().__init__()
        self.encoder = nn.Linear(inp, out, bias=False)
        self.decoder = nn.Linear(out, inp, bias=False)

        # tie the weights
        #print(type(self.encoder.weight))
        self.encoder.weight = nn.Parameter(weight)
        self.decoder.weight = nn.Parameter(weight.transpose(0,1))

    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = self.decoder(encoded_feats)
        return encoded_feats, reconstructed_output

# tied auto encoder using functional calls
class TiedAutoEncoderFunctional(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.param = nn.Parameter(torch.randn(out, inp))

    def forward(self, input):
        encoded_feats = F.linear(input, self.param)
        reconstructed_output = F.linear(encoded_feats, self.param.t())
        return encoded_feats, reconstructed_output

# mixed approach
class MixedAppraochTiedAutoEncoder(nn.Module):
    def __init__(self, inp, out, weight):
        super().__init__()
        self.encoder = nn.Linear(inp, out, bias=False)
        self.encoder.weight = nn.Parameter(weight)

    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = F.linear(encoded_feats, self.encoder.weight.t())
        return encoded_feats, reconstructed_output

if __name__ == '__main__':
    tied_module_F = TiedAutoEncoderFunctional(5, 6)

    # instantiate off-the-shelf auto-encoder
    offshelf_weight = tied_module_F.param.data.clone()
    tied_module_offshelf = TiedAutoEncoderOffTheShelf(5, 6, offshelf_weight)

    # instantiate mixed type auto-encoder
    mixed_weight = tied_module_F.param.data.clone()
    tied_module_mixed = MixedAppraochTiedAutoEncoder(5, 6, mixed_weight)

    assert torch.equal(tied_module_offshelf.encoder.weight.data, tied_module_F.param.data), 'F vs offshelf: param not equal'
    assert torch.equal(tied_module_mixed.encoder.weight.data, tied_module_F.param.data), 'F vs mixed: param not equal'

    optim_F = optim.SGD(tied_module_F.parameters(), lr=1)
    optim_offshelf = optim.SGD(tied_module_offshelf.parameters(), lr=1)
    optim_mixed = optim.SGD(tied_module_mixed.parameters(), lr=1)
    # common input
    input = torch.rand(5, 5)

    # zero the gradients
    optim_F.zero_grad()
    optim_offshelf.zero_grad()
    optim_mixed.zero_grad()

    # get output from both modules      
    reconstruction_F = tied_module_F(input) 
    reconstruction_offshelf = tied_module_offshelf(input) 
    reconstruction_mixed = tied_module_mixed(input) 

    # back propagation
    reconstruction_F[1].sum().backward()
    reconstruction_offshelf[1].sum().backward()
    reconstruction_mixed[1].sum().backward()

    # step
    optim_F.step()
    optim_offshelf.step()
    optim_mixed.step()

    # check the equality of output and parameters
    assert torch.equal(reconstruction_offshelf[0], reconstruction_F[0]), 'F vs offshelf: bottleneck not equal'
    assert torch.equal(reconstruction_offshelf[1], reconstruction_F[1]), 'F vs offshelf: output not equal'
    assert (tied_module_offshelf.encoder.weight.data - tied_module_F.param.data).pow(2).sum() < 1e-10, 'F vs offshelf: param after step not equal'
    assert (tied_module_offshelf.encoder.weight.data - offshelf_weight).pow(2).sum() < 1e-10, 'F vs mixed: source weight tensor not equal'

    assert torch.equal(reconstruction_mixed[0], reconstruction_F[0]), 'F vs mixed: bottleneck not equal'
    assert torch.equal(reconstruction_mixed[1], reconstruction_F[1]), 'F vs mixed: output not equal'
    assert (tied_module_mixed.encoder.weight.data - tied_module_F.param.data).pow(2).sum() < 1e-10, 'F vs mixed: param after step not equal'
    assert (tied_module_mixed.encoder.weight.data - mixed_weight).pow(2).sum() < 1e-10, 'F vs mixed: param after step not equal'

    print('success!')