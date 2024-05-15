import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, stats=[], name='AlexNet with self-attention'):
        self.stats = stats
        self.name = name
        self.axes = self.training_plot()

    def training_plot(self):
        print('Setting up interactive graph...')
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(self.name)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('x Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('y Loss')
        return axes


    def update_plot(self, epoch):
        xrange = range(epoch - len(self.stats) + 1, epoch + 1)
        self.axes[0].plot(xrange, [s[0] for s in self.stats], linestyle='--', marker='o', color='b')
        self.axes[0].plot(xrange, [s[2] for s in self.stats], linestyle='--', marker='o', color='r')
        self.axes[1].plot(xrange, [s[1] for s in self.stats], linestyle='--', marker='o', color='b')
        self.axes[1].plot(xrange, [s[3] for s in self.stats], linestyle='--', marker='o', color='r')
        #self.axes[0].legend(['Validation', 'Train'])
        #self.axes[1].legend(['Validation', 'Train'])
        #self.axes.plot(xrange, [s[0] for s in self.stats], linestyle='--', marker='o', color = 'r')
        #self.axes.plot(xrange, [s[1] for s in self.stats], linestyle='--', marker='o', color = 'b')
        self.axes[0].legend(['Validation', 'Train'])
        self.axes[1].legend(['Validation', 'Train'])
        plt.pause(0.00001)

    def save_plot(self):
        plt.savefig(self.name + '_training_plot.png', dpi=200)

    def hold_plot(self):
        plt.ioff()
        plt.show()
