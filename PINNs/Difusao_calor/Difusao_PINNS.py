import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

'''
Este programa tem como objetivo determinar a distribuição da temperatura em função da posição 
x e do tempo (t) em uma placa plana. Inicialmente com temperatura uniforme (Ti), que é imersa 
em um fluido com temperatura T_inifito. Para isso, foram utilizadas Redes Neurais Baseadasem 
Física, conhecidas como PINNs (Physics-Informed Neural Networks).
'''

# Parâmetros do problema
alpha = 18.8 * 1e-6  # Difusividade térmica
h = 500.  # Coeficiente de transferência de calor convectivo
k = 63.9  # Condutividade térmica
T_infinito = 333.15  # Temperatura ambiente
Ti = 253.15  # Temperatura inicial da placa
L = 40 * 1e-3  # Meio comprimento da placa na direçao x
tempo_final = 480  # Tempo máximo

# Definindo uma seed para tornar os resultados reproduzíveis
torch.manual_seed(42)
np.random.seed(42)

# Definição da rede neural
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.initialize_weights()

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

# Função de perda (Loss Function = MSE_Condição_Inicial + λ1 * MSE_EDP + λ2 * MSE_Condiçoes_de_Contorno)
flag = False
def pinn_loss(model: nn.Module, x: torch.Tensor, tempo: torch.Tensor):
    global flag
    
    # Condição inicial θ(x*, 0) = 1
    T_inicial = model(torch.cat([x, torch.zeros_like(tempo)], dim=1))
    init_loss = torch.mean((T_inicial - 1) ** 2)
    
    # Condição de contorno: dθ/dx* = -biot*(θ(1, t*)) em x* = 1
    x_borda = torch.ones_like(x)  # Redefinindo x para a borda
    x_borda.requires_grad_(True)  # Certificando-se que x_borda tem gradientes
    T_borda = model(torch.cat([x_borda, tempo], dim=1))  
    dT_dx_borda = torch.autograd.grad(T_borda, x_borda, grad_outputs=torch.ones_like(T_borda), create_graph=True)[0]
    boundary_loss = torch.mean((dT_dx_borda + h*L/k*(T_borda)) ** 2)

    # Condição de contorno: dθ/dx* = 0 em x* = 0
    x_borda = torch.zeros_like(x)  # Redefinindo x para a borda
    x_borda.requires_grad_(True)  # Certificando-se que x_borda tem gradientes
    T_borda = model(torch.cat([x_borda, tempo], dim=1))  
    dT_dx_borda = torch.autograd.grad(T_borda, x_borda, grad_outputs=torch.ones_like(T_borda), create_graph=True)[0]
    boundary_loss2 = torch.mean(dT_dx_borda ** 2)

    #teste
    T_final = model(torch.cat([torch.ones_like(x), tempo], dim=1))
    final_loss = torch.mean((T_final)**2)

    #Função de perda sem a EDP
    loss_function = final_loss

    #Somente calcula a funçao de perda da EDP apos atingir um erro menor que 2x10^-5
    if flag or loss_function < 2e-5:
        x.requires_grad_(True)
        tempo.requires_grad_(True)

        # EDP d²θ/dx*² = L²/(tempo_final*alpha)*dθ/dt*
        temperatura = model(torch.cartesian_prod(x.squeeze(1), tempo.squeeze(1)))
        dT_dx = torch.autograd.grad(temperatura, x, grad_outputs=torch.ones_like(temperatura), create_graph=True)[0]
        dT_dt = torch.autograd.grad(temperatura, tempo, grad_outputs=torch.ones_like(temperatura), create_graph=True)[0]  
        d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
        eq_loss = torch.mean((d2T_dx2 - L**2/(tempo_final*alpha)*dT_dt) ** 2)

        loss_function = init_loss + boundary_loss + boundary_loss2 + eq_loss

        flag = True

    return loss_function
    
# Função de treinamento
def train(model, optimizer, num_points=200):
    tempo = torch.linspace(0, 1, num_points).unsqueeze(1)
    x = torch.linspace(0, 1, num_points).unsqueeze(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss = pinn_loss(model, x, tempo)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {current_lr}')

    # Avaliando o modelo treinado
    temperatura_pred = model(torch.cat([x, tempo], dim=1))
    return temperatura_pred

# Criando o modelo e otimizador
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10000)

# Treinamento
epochs = 15000
temperatura_pred = train(model, optimizer)

#Salvando o modelo
torch.save(model.state_dict(), "modelo_treinado.pth")

# Obtendo os valores de temperatura previstos
x_flat = torch.linspace(0, 1, 300)
tempo_flat = torch.linspace(0, 1, 300)

#Criando a matriz com os valores da Temperatura prevista pelo modelo
matrix_temperatura = np.array([]).reshape(0, len(x_flat)) 
for x in x_flat: 
    tensor = torch.stack([x*torch.ones_like(x_flat), tempo_flat], dim=1)
    temperatura_pred = model(tensor).detach().numpy()
    matrix_temperatura = np.vstack([matrix_temperatura, temperatura_pred.reshape(-1)])

temperatura_pred = matrix_temperatura*(Ti-T_infinito) + T_infinito
temperatura_pred = np.transpose(temperatura_pred)
print(temperatura_pred)
temperatura_pred = np.hstack((temperatura_pred[:, ::-1], temperatura_pred[:, 1:]))

#Grafico de calor da funçao: T(x, t) para x ∈ [-L, L] e t ∈ [0, tempo_maximo]
plt.figure(figsize=(8, 6))
plt.imshow(temperatura_pred[::-1], cmap='coolwarm', extent=[-L, L, 0, tempo_final], aspect='auto')
plt.colorbar(label='Temperatura')
plt.xlabel('Posição [m]')
plt.ylabel('Tempo [s]')
plt.title('PINNs: T(x, t)')
plt.show()
