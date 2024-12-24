import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

'''
This program aims to determine the temperature distribution as a function of position (x) 
and time (t) in a flat plate. Initially at a uniform temperature (T_i), the plate is
immersed in a fluid with temperature (T_infty). To achieve this, Physics-Informed 
Neural Networks (PINNs) were used.
'''

torch.manual_seed(42)
np.random.seed(42)

#---------------------------------Parameters ------------------------------------------------------

alpha = 18.8 * 1e-6     # Thermal Diffusivity
h = 500.                # Convective Heat Transfer Coefficient
k = 63.9                # Thermal Conductivity
T_infty = 333.15        # Fluid Temperature
Ti = 253.15             # Initial Temperature of the Plate Temperatura
L = 40 * 1e-3           # bar Length
t_end = 480             # End Time

class PINN(nn.Module):
    def __init__(self, input_shape: int, num_layers: int, hidden_units: int, output_shape: int):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_shape, hidden_units))
        layers.append(nn.Tanh())

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_units, output_shape))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def loss_function(model: nn.Module, x: torch.Tensor, t: torch.Tensor):

    x.requires_grad_(True)
    t.requires_grad_(True)
    input = torch.concat([x, t], dim=1)
    u = model(input)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    #Partial Differential Equation: d²θ/dx*² = L²/(tempo_final.alpha).dθ/dt*
    loss_PDE = torch.mean((u_xx - L**2/(t_end*alpha)*u_t)**2)

    #Boundary Condition left: dθ/dx* = 0 at x* = 0
    mask = (x == 0)
    loss_BC = torch.mean((u_x[mask])**2)

    #Boundary Condition right: dθ/dx* = -biot.(θ(1, t*)) at x* = 1
    mask = (x == 1)
    loss_BC += torch.mean((u_x[mask] + h*L/k*u[mask])**2)

    #Initial Condition:  θ(x*, 0) = 1 at t* = 0
    mask = (t == 0)
    loss_IC = torch.mean((u[mask] - 1))**2

    return 0.1*loss_PDE + loss_BC + loss_IC

#Creating the model and optimizer
model = PINN(2, 5, 20, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training
x = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)

X, T = torch.meshgrid(x, t, indexing="ij")
X = X.flatten().view(-1, 1)
T = T.flatten().view(-1, 1)

epochs = 4000
model.train()
for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    loss = loss_function(model, X, T)
    loss.backward()
    optimizer.step()

    if epoch % 250 == 0:
        tqdm.write(f"Epoch: {epoch} || loss: {loss.item()}")

#Graphic
model.eval()
X_np, T_np = np.meshgrid(x.numpy(), t.numpy())
with torch.no_grad():
    entrada = torch.tensor(np.stack([X_np, T_np], axis=-1), dtype=torch.float32)
    u = model(entrada).numpy().squeeze()
    u = u*(Ti - T_infty) + T_infty

#Mirror
X_np = np.concatenate([X_np, -X_np[::-1]])
T_np = np.concatenate([T_np, T_np[::-1]])
u = np.concatenate([u, u[::-1]])


plt.contourf(1e3*L*X_np, t_end*T_np, u, 100, cmap="hot")
plt.colorbar(label="Temperatura")
plt.xlabel("x [Comprimento]")
plt.ylabel("t [tempo]")
plt.show()