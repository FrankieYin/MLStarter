import network, mnist_loader, time
import matplotlib.pyplot as plt

def main():
    train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 1000, 10])
    net.sgd(train_data, 30, 10, 0.5,
            lmbda=5.0,
            momentum_coefficient=0.1,
            test_data=test_data,
            dropout_enabled=False,
            l2_enabled=True,
            early_stopping=False)

    plt.title("done")
    plt.show()

if __name__ == "__main__":
    main()
