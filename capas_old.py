import numpy as np

np.random.seed(312007)

class Layer():
    def __init__(self, size: int):
        self.size = size
        self.output = None


class LayerInput(Layer):
    def __init__(self, size: int):
        super().__init__(size)

    def valuesasing(self, values):
        out = np.array(values)
        self.output = out.transpose()


class LayerFullyConnected(Layer):
    def __init__(self, size: int, input: Layer):
        super().__init__(size)
        self.input = input
        self.bias = None
        self.weight = None
        self.initialization()

    def initialization(self):
        self.bias = np.zeros(self.size)
        self.weight = np.random.normal(0, 1, [self.size, self.input.size])
        # print(self.bias)
        # print(self.weight.shape)

    def forward(self):
        self.output = np.dot(self.weight, self.input.output)  # + self.bias


class LayerOutput(Layer):
    def __init__(self, size: int):
        super.__init__(size)


culoLayer = LayerInput(size=2)
culoLayer.valuesasing([3, 5])

print(culoLayer.output)

culo = LayerFullyConnected(size=5, input=culoLayer)
culo.forward()
print(culo.weight)
print("Culo output:" ,culo.output)