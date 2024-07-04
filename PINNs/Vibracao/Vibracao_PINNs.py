import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definindo uma seed para tornar os resultados reproduzíveis
torch.manual_seed(42)
np.random.seed(42)

# Definindo os parâmetros do sistema --------------------------------------------------------------
m = 1.0 # Massa em kg
k = 4.0 # Rigidez da mola em N/m
mu = 0.3 # Coeficiente de amortecimento em N.s/m

# Definindo a condição inicial e condição de contorno
x0 = 1.0 # Posição inicial em m
v0 = 0.0 # Velocidade inicial em m/s
tempo_final = 10

# Estrutura da Rede Neural ------------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.camada1 = nn.Linear(1, 64)
        self.camada2 = nn.Linear(64, 64)
        self.camada3 = nn.Linear(64, 64)
        self.camada4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.camada1(x))
        x = torch.tanh(self.camada2(x))
        x = torch.tanh(self.camada3(x))
        return self.camada4(x)
    
# Função de Perda
flag = True
def loss_function(model) -> torch.Tensor:
    global flag

    # Condiçao Inicial x(0) = 1
    t_inicial = torch.tensor(0.).reshape(-1, 1).requires_grad_(True)
    x_inicial = model(t_inicial)
    loss_inicial = (x_inicial - x0)**2

    # Condição de contorno de Neumann dx/dt = 0 em t = 0
    dxdt_boundary = torch.autograd.grad(x_inicial, t_inicial, grad_outputs=torch.ones_like(x_inicial), create_graph=True)[0]
    loss_boundary = (dxdt_boundary - v0)**2

    # No começo do treinamento o modelo foca apenas nas condiçoes iniciais e de contorno
    if flag and (loss_inicial + loss_boundary < 0.01):
        return loss_inicial + loss_boundary
    flag = False

    # EDO
    t = torch.linspace(0, tempo_final, 1000).reshape(-1, 1).requires_grad_(True)
    x = model(t)
    dxdt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dx2dt2 = torch.autograd.grad(dxdt, t, grad_outputs=torch.ones_like(dxdt), create_graph=True)[0]
    loss_EDO = torch.mean((m*dx2dt2 + mu*dxdt + k*x)**2)

    return loss_inicial + loss_boundary + loss_EDO

# Criando o modelo a partir da minha rede neural estruturada
modelo = PINN()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

#Treinando
epochs = 15000
for _ in range(epochs):
    otimizador.zero_grad()
    loss = loss_function(modelo)
    loss.backward()
    otimizador.step()

# Resultados do modelo treinado
tempo = torch.linspace(0, tempo_final, 10001).reshape(-1, 1)
x_previsto = modelo(tempo).detach()

# Grafico x(t) a partir dos resultado do modelo
plt.plot(tempo.numpy().squeeze(), x_previsto.numpy().squeeze(), label = "PINNs")
plt.xlabel("Tempo [s]")
plt.ylabel("Posição [m]")
plt.show()