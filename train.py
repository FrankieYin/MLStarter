import network, mnist_loader

def main():
    train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.sgd(train_data, 30, 10, 0.5, lmbda=5.0, test_data=test_data)

if __name__ == "__main__":
    main()
