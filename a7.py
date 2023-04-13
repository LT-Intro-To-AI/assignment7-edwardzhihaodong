from neural import NeuralNet

XOR = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

nn = NeuralNet(2, 3, 1)

nn.train(XOR)

print(nn.test_with_expected(XOR))
print()
print(nn.get_ih_weights())
print()
print(nn.get_ho_weights())

XOR2 = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

nn2 = NeuralNet(2,8,1)

nn2.train(XOR2)

print(nn.get_ih_weights())
print()
print(nn.get_ho_weights())