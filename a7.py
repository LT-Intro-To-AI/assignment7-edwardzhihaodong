from neural import *

XOR = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

nn = NeuralNet(2, 2, 1)

nn.train(XOR)

print(nn.get_ih_weights())
print()
print(nn.get_ho_weights())