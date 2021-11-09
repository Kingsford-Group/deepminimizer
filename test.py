from util import *
from matplotlib import pyplot as plt

pos = torch.arange(100).view(-1, 1) * 1.0
net = nn.Sequential(
    nn.Linear(1, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.ReLU()
)

val = net(torch.cos(pos))

pos = pos.view(-1).detach().numpy()
val = val.view(-1).detach().numpy()
print(pos)
print(val)
plt.figure()
plt.plot(pos, val)
plt.savefig('./test.png')