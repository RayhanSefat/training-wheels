# %% 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from .nn_blocks import Linear, ReLU

rc('animation', html='jshtml')

X = torch.linspace(-5, 5, 100).view(-1, 1)
Y = X * X

class VanillaNN(nn.Module):
    def __init__(self):
        super(VanillaNN, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1, 8))
        self.w2 = nn.Parameter(torch.randn(8, 16))
        self.w3 = nn.Parameter(torch.randn(16, 1))
        self.b1 = nn.Parameter(torch.randn(8))
        self.b2 = nn.Parameter(torch.randn(16))
        self.b3 = nn.Parameter(torch.randn(1))
        self.linear1 = Linear(1, 8)
        self.linear1.load_state_dict({"weight": self.w1.T, "bias": self.b1})
        self.linear2 = Linear(8, 16)
        self.linear2.load_state_dict({"weight": self.w2.T, "bias": self.b2})
        self.linear3 = Linear(16, 1)
        self.linear3.load_state_dict({"weight": self.w3.T, "bias": self.b3})

    def forward(self, x):
        x = ReLU(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

snapshots = []
model = VanillaNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10001):
    Y_hat = model(X)
    loss = torch.mean((Y_hat - Y) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        snapshots.append(Y_hat.detach().numpy())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

fig, ax = plt.subplots()
ax.plot(X.numpy(), Y.numpy(), 'r--', alpha=0.5, label='Target: $y=x^2$')
line, = ax.plot([], [], 'b-', lw=2, label='NN Prediction')
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 27)
ax.legend()

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(X.numpy(), snapshots[i])
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(snapshots), interval=50, blit=True
)

ani.save('therapml/animation.gif', writer='pillow', fps=500)