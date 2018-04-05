import network, mnist_loader, time
import matplotlib.pyplot as plt

def main():
    train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    ns = [50, 70, 90, 100, 200, 400, 600, 800, 1000]
    d = dict()
    for n in ns:
        k = "784-{}-10".format(n)
        net = network.Network([784, n, 10])
        accuracy, _, _, _ = net.sgd(train_data, 30, 10, 0.5,
                lmbda=5.0,
                momentum_coefficient=0.1,
                test_data=test_data,
                dropout_enabled=True,
                l2_enabled=False,
                early_stopping=False)
        d[k] = accuracy

    print(d)
    # plt.title("done")
    # plt.show()

if __name__ == "__main__":
    main()
