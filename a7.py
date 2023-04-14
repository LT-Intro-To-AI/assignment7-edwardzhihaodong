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

nn2 = NeuralNet(2,8,1)

nn2.train(XOR)

print(nn2.test_with_expected(XOR))
print()
print(nn2.get_ih_weights())
print()
print(nn2.get_ho_weights())

nn3 = NeuralNet(2, 1, 1)

nn3.train(XOR)

print(nn3.test_with_expected(XOR))
print()
print(nn3.get_ih_weights())
print()
print(nn3.get_ho_weights())

print("\nTraing Voter Opinions Now \n")

voter_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])
]

von = NeuralNet(5, 20, 1)

von.train(voter_data)
print(von.test_with_expected(voter_data))

print(f"case 1: {von.evaluate([1,1,1,1,1])}")

voter_testing = [
    [1, 1, 1, 0.1, 0.1],
    [0.5, 0.2, 0.1, 0.7, 0.7],
    [0.8, 0.3, 0.3, 0.3, 0.8],
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6],
]

print(von.evaluate(voter_testing[0]))
print(von.evaluate(voter_testing[1]))
print(von.evaluate(voter_testing[2]))
print(von.evaluate(voter_testing[3]))
print(von.evaluate(voter_testing[4]))