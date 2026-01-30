import torch
from torch import nn

from src.dataset import get_linear_data
from src.train import train_gan

data = get_linear_data()
dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data),
        batch_size=8, shuffle=True)

latent_dim = 2
input_dim = data.shape[1]

generator = nn.Sequential(
    nn.Linear(latent_dim, input_dim)
)

discriminator = nn.Sequential(
    nn.Linear(input_dim, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1)
)

lr_D = 0.05
lr_G = 0.005
num_epochs = 20
D_trainer = torch.optim.Adam(discriminator.parameters(), 0.05)
G_trainer = torch.optim.Adam(generator.parameters(), 0.005)
criterion = nn.BCEWithLogitsLoss(reduction="sum")
num_epochs = 30
dataloader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(data),
    batch_size=16, shuffle=True
)
fixed_noise = torch.normal(0, 1, size=(100, latent_dim))
train_gan(discriminator, generator, dataloader, data[:100], num_epochs, lr_G, lr_D, latent_dim, visualize=True, print_every=25, fixed_noise=fixed_noise)