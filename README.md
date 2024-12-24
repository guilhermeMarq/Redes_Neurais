# Physics-Informed Neural Networks (PINNs)

## Descrição
Este projeto é uma implementação em Python de Redes Neurais Informadas pela Física (PINNs), destinada a resolver três problemas distintos de engenharia e física. O programa é destinado a modelar sistemas fisicos através de equações diferenciais, condições iniciais e condições de contorno que governam o fenômeno físico estudado.

## Problemas Abordados
1. **Sistema Massa-Mola-Amortecedor**
2. **Deformação de Material Heterogêneo**
3. **Difusão de Calor**

### 1. Sistema Massa-Mola-Amortecedor
O primeiro problema visa modelar um sistema de vibração livre de um grau de liberdade com amortecimento viscoso. A equação do movimento que descreve o comportamento dinâmico do sistema é dada por:
$$m \frac{d^2 x}{dt^2} + \mu \frac{dx}{dt} + kx = 0$$
![ezgif com-added-text](https://github.com/guilhermeMarq/Redes-Neurais/assets/72332375/2b8a5f42-80d4-46de-ac4f-07530bcf3074)

### 2. Deformação de uma barra heterogênea
O segundo problema consiste em determina a deformação $u$ ao longo de uma barra elástica de comprimento $L$ composta por dois materiais diferentes, com módulo de elasticidade $E_1$ e $E_2$, respectivamente, e comprimentos $L_1$ e $L_2$. A equação geral de equilíbrio de forças em um corpo elástico submetido a um carregamento axial unidimensional na direção $x$ com modolu de elasticidade variavel pode ser expressa como:
$$\frac{d}{dx}\left[E(x)\frac{du}{dx}\right] = 0$$

$$
E(x) = \begin{cases}
E_1 & \text{para } 0 \leq x \leq L_1 \\
E_2 & \text{para } L_1 \leq x \leq L
\end{cases}
$$

### 3. Difusão de Calor
O terceiro problema tem como objetivo modelar a distribuição de temperatura em um sistema com transferência de calor unidimensional para coordenadas cartesianas, em regime transiente e sem geração de calor. A equação diferencial parcial que descreve esse sistema é igual a:
$$k \frac{\partial^2 T}{\partial x^2} = \rho c_p \frac{\partial T}{\partial t}$$

![Grafico 3d - Soluçao Analitica](https://github.com/guilhermeMarq/Redes-Neurais/assets/72332375/e0882124-0010-4198-9651-efc1c5f0ef89)



## Como Usar
Para utilizar este programa, siga os passos abaixo:
1. Clone o repositório para sua máquina local.
2. Instale as dependências necessárias utilizando `pip install -r requirements.txt`.
3. Execute o script principal para cada um dos problemas através do comando `python main.py --problema [nome_do_problema]`.

## Contribuições
Contribuições são sempre bem-vindas. Para contribuir, por favor, crie um fork do repositório, faça suas alterações e submeta um pull request para avaliação.

## Licença
Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

OBS: Este README oferece uma visão geral do projeto e sera atualizado conforme o desenvolvimento progride e novas funcionalidades são implementadas.
