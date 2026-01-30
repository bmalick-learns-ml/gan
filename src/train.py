import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

def update_discriminator(x, z, D, G, criterion, trainer_D):
    """Update discriminator."""
    batch_size = x.shape[0]
    ones = torch.ones((batch_size,), device=x.device)
    zeros = torch.zeros((batch_size,), device=x.device)

    trainer_D.zero_grad()

    real_y = D(x)
    fake_y = D(G(z))
    loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
                          criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_generator(z, D, G, criterion, trainer_G):
    """Update generator."""
    batch_size = z.shape[0]
    ones = torch.ones((batch_size,), device=z.device)

    trainer_G.zero_grad()

    fake_y = D(G(z))
    loss_G = criterion(fake_y, ones.reshape(fake_y.shape))
    loss_G.backward()
    trainer_G.step()
    return  loss_G

def train_gan(D, G, dataloader, data, num_epochs, lr_G, lr_D, latent_dim, fixed_noise, visualize=False, print_every=25):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    #for w in D.parameters(): nn.init.normal_(w, 0., 0.02)
    #for w in G.parameters(): nn.init.normal_(w, 0., 0.02)
    trainer_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(G.parameters(), lr=lr_G)

    metrics = []
    os.makedirs("visualizations", exist_ok=True)
    for epoch in range(num_epochs):
        loss_D = 0
        loss_G = 0
        num_instances = 0
        for step_num, (X,) in enumerate(dataloader):
            batch_size = X.shape[0]
            num_instances += batch_size
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            loss_D += update_discriminator(X, Z, D, G, loss, trainer_D).item() * batch_size
            loss_G += update_generator(Z, D, G, loss, trainer_G).item() * batch_size
            # if step_num % print_every == 0:
            #     print(f"[Epoch {epoch+1}/{num_epochs}] [Step {step_num}/{len(dataloader)}] loss_D: {loss_D/num_instances:.5f}, loss_G: {loss_G/num_instances:.5f}")

        loss_G /= num_instances
        loss_D /= num_instances
        metrics.append([loss_D, loss_G])
        print(f"[Epoch {epoch+1}/{num_epochs}] loss_D: {loss_D:.5f}, loss_G: {loss_G:.5f}")

        # Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_data = G(fixed_noise).detach().numpy()
        plt.scatter(data[:, 0], data[:, 1], label="real")
        plt.scatter(fake_data[:, 0], fake_data[:, 1], label="generated")
        plt.legend(["real", "generated"])
        plt.title(f"Epoch {epoch:02d}")
        plt.savefig(f"visualizations/{epoch:02d}.png")
        # plt.show()
        plt.close()

    metrics = np.array(metrics)

    plt.plot(metrics[:, 0], label="discriminator")
    plt.plot(metrics[:, 1], label="generator", linestyle="--")
    plt.legend()
    plt.ylabel("loss")
    plt.show(block=False)
    plt.pause(3)
    plt.close()