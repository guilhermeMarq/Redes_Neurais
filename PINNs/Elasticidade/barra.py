import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

#Parâmetros (SI)
F = 10 #Força externa
L = 1. #comprimento da barra

#Modulo de elasticidade
L1 = 0.5
E1 = 10
E2 = 5

def solucao_analitica(x: np.ndarray) -> np.ndarray:
    n = int(x.size/2)

    u = np.zeros_like(x)
    u[:n] = F*x[:n]/E1
    u[n:] = F*x[n:]/E2 + F*L1*(1/E1-1/E2)
    return u

def erro_absoluto(x, u):
    u_exata = solucao_analitica(x)
    return np.mean(np.square(u - u_exata))

# Definindo uma seed para tornar os resultados reproduzíveis
torch.manual_seed(42)
np.random.seed(42)

#Estrutura da rede neural
class PINNs(nn.Module):
    def __init__(self):
        super().__init__()
        self.camada1 = nn.Linear(1, 64)
        self.camada2 = nn.Linear(64, 64)
        self.camada3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.camada1(x))
        x = torch.tanh(self.camada2(x))
        x = self.camada3(x)
        return x
    
#Função da perda
def funcao_perda(model) -> torch.Tensor:

    #Condição inicial u(0) = 0
    x_inicial = torch.tensor(0.).reshape(-1, 1).requires_grad_(True)
    u_inicial = model(x_inicial)
    perda_inicial = (u_inicial)**2

    #Condiçao de contorno E1*du/dx = F/A
    x_borda = torch.linspace(0., L1, 200).reshape(-1, 1).requires_grad_(True)
    u_borda = model(x_borda)
    dudx_borda = torch.autograd.grad(u_borda, x_borda, grad_outputs=torch.ones_like(u_borda), create_graph=True)[0]
    perda_contorno = torch.mean((dudx_borda - F/E1)**2)

    #Condiçao de contorno E2*du/dx = F/A
    x_borda_2 = torch.linspace(L1, L, 200).reshape(-1, 1).requires_grad_(True)
    u_borda_2 = model(x_borda_2)
    dudx_borda_2 = torch.autograd.grad(u_borda_2, x_borda_2, grad_outputs=torch.ones_like(u_borda_2), create_graph=True)[0]
    perda_contorno_2 = torch.mean((dudx_borda_2 - F/E2)**2)


    return 10*perda_inicial + perda_contorno + perda_contorno_2

#Modelo e o otimizador
modelo = PINNs()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

#Treino
interaçoes = 7001
for i in range(interaçoes):
    otimizador.zero_grad()
    perda = funcao_perda(modelo)
    perda.backward()
    otimizador.step()

    if i % 1000 == 0:
        print(f'Epoch {i}, Loss: {perda.item()}')

#Resultado do modelo (PINNS)
x_prev = torch.linspace(0, L, 300).reshape(-1, 1)
u_prev = modelo(x_prev).detach()

#Resultado analitico
x = np.linspace(0, L, 300)
u = solucao_analitica(x)

#Grafico
#plt.plot(x_prev.numpy().squeeze(), u_prev.numpy().squeeze(), label="PINNs")
plt.plot(x, u, label="Solução analitica")
plt.plot(x, homogenizacao(x), label="Homogenização")
plt.ylabel("Deformação [m/m]")
plt.xlabel("Posição [m]")
plt.legend()
plt.show()