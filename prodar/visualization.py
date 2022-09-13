from os import path
import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    def __init__(self, save_dir='.'):

        self.save_dir = save_dir

        self.pr_num = 0

    def plot_pr(self, precision, recall, name=None,
                hold=False, filename_suffix=None):

        fig = plt.figure('pr', figsize=(4,4), dpi=300, constrained_layout=True)
        ax = fig.gca()

        if self.pr_num == 0:

            # f1 contour

            levels = 10

            spacing = np.linspace(0, 1, 1000)
            x, y = np.meshgrid(spacing, spacing)

            with np.errstate(divide='ignore', invalid='ignore'):
                f1 = 2 / (1/x + 1/y)

            locx = np.linspace(0, 1, levels, endpoint=False)[1:]

            cs = ax.contour(x, y, f1, levels=levels, linewidths=1, colors='k',
                            alpha=0.3)
            ax.clabel(cs, inline=True, fmt='F1=%.1f',
                      manual=np.tile(locx,(2,1)).T)

        with np.errstate(divide='ignore', invalid='ignore'):
            aupr = np.trapz(np.flip(precision), x=np.flip(recall))
            f1 = 2*recall*precision / (recall+precision)
        f1_max_idx = np.nanargmax(f1)
        f1_max = f1[f1_max_idx]

        ax.plot(recall, precision, lw=1, color=f'C{self.pr_num}', label=name)

        ax.plot(recall[f1_max_idx], precision[f1_max_idx], marker='o',
                mfc='none', mec=f'C{self.pr_num}', mew=0.5, ms=5)

        if not hold:
            plt.xlabel('recall')
            plt.ylabel('precision')
            if self.pr_num == 0:
                # plt.title(f'AUPR: {aupr}, f1: {f1_max}')
                plt.title(f'f1$_{{max}}$: {f1_max:.6f}')

                # plt.show()
                filename = (f'pr_curve-{filename_suffix}.png'
                                if filename_suffix else 'pr_curve.png')
                plt.savefig(path.join(self.save_dir, filename))
            else:
                plt.legend()
                self.pr_num = 0

            plt.clf()
        else:
            self.pr_holding = True
            self.pr_num += 1

    def plot_loss_acc_hist(self, loss_acc_hist, filename_suffix=None):

        n_epochs = loss_acc_hist.shape[0]

        # plot loss
        fig = plt.figure('loss', figsize=(6,4), dpi=300,
                         constrained_layout=True)
        plt.plot(np.arange(1, n_epochs+1), loss_acc_hist[:,0], label='train')
        plt.plot(np.arange(1, n_epochs+1), loss_acc_hist[:,1], label='valid')
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.xlim(0, n_epochs)
        plt.legend()
        plt.grid()
        filename = 'training_hist-loss.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.clf()

        # plot accuracy
        fig = plt.figure('acc', figsize=(6,4), dpi=300,
                         constrained_layout=True)
        plt.plot(np.arange(1, n_epochs+1), loss_acc_hist[:,2], label='train')
        plt.plot(np.arange(1, n_epochs+1), loss_acc_hist[:,3], label='valid')
        plt.xlabel('epoch number')
        plt.ylabel('accuracy')
        plt.xlim(0, n_epochs)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        filename = 'training_hist-acc.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.clf()

    def plot_f1_max_hist(self, f1_max_hist, filename_suffix=None):

        n_epochs = f1_max_hist.shape[0]

        # plot loss
        fig = plt.figure('f1', figsize=(6,4), dpi=300,
                         constrained_layout=True)
        plt.plot(np.arange(1, n_epochs+1), f1_max_hist[:,0],
                 label='f1$_{max}$')
        plt.plot(np.arange(1, n_epochs+1), f1_max_hist[:,1],
                 label='precision @ f1$_{max}$', alpha=0.7)
        plt.plot(np.arange(1, n_epochs+1), f1_max_hist[:,2],
                 label='recall @ f1$_{max}$', alpha=0.7)
        plt.xlabel('epoch number')
        plt.ylabel('f1$_{max}$/precision/recall')
        plt.xlim(0, n_epochs)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        filename = 'training_hist-f1_max.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.clf()


if __name__ == '__main__':

    plotter = Plotter()
    plotter.plot_pr()
