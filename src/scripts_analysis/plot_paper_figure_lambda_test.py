'''
Unfortunately this is not automated enough. The values are manually entered rather than read from the files.
'''
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lambda_contrastive = [0, 0.001, 0.01, 0.03, 0.1, 0.5, 0.9, 1]
    dice_mean = [0.682, 0.682, 0.690, 0.634, 0.550, 0.546, 0.502, 0.471]
    dice_sem = [0.028, 0.029, 0.028, 0.030, 0.028, 0.031, 0.028, 0.047]
    ssim_mean = [0.875, 0.873, 0.879, 0.857, 0.825, 0.817, 0.809, 0.748]
    ssim_sem = [0.007, 0.008, 0.007, 0.010, 0.008, 0.010, 0.009, 0.031]
    ergas_mean = [4614, 4491, 4723, 4866, 6033, 6256, 6024, 6208]
    ergas_sem = [247, 248, 262, 260, 285, 288, 253, 576]
    rmse_mean = [0.212, 0.213, 0.208, 0.228, 0.269, 0.247, 0.273, 0.304]
    rmse_sem = [0.010, 0.010, 0.010, 0.010, 0.009, 0.011, 0.008, 0.023]

    log_x = np.log(np.array(lambda_contrastive) + 1e-3)
    # Spacing out for better display.
    for i in range(len(log_x)):
        log_x[i] += 5e-1 * i

    xticklabels = [
        '0\n\n\u03BB = 0\nNo Contrastive',
        '0.001',
        '0.01',
        '0.03',
        '0.1',
        '0.5',
        '0.9',
        '1\n\n\u03BB = 1\nNo Reconstruction',
    ]
    xticklabel_rotation = [30, 0, 0, 0, 0, 0, 0, 30]

    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    fig = plt.figure(figsize=(22, 5))
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(log_x, dice_mean, c='darkblue', linewidth=2)
    ax.fill_between(log_x,
                    np.array(dice_mean) - np.array(dice_sem),
                    np.array(dice_mean) + np.array(dice_sem),
                    color='darkblue',
                    alpha=0.5)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=log_x[np.argmax(dice_mean)],
              ymin=ymin,
              ymax=dice_mean[np.argmax(dice_mean)],
              colors='k',
              linestyles='dashed')
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(log_x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('\nWeighting Coefficient \u03BB (Log Scale)', fontsize=12)
    ax.set_ylabel('Dice\n(HIGHER is better)', fontsize=12)

    ax = fig.add_subplot(1, 4, 2)
    ax.plot(log_x, ssim_mean, c='darkblue', linewidth=2)
    ax.fill_between(log_x,
                    np.array(ssim_mean) - np.array(ssim_sem),
                    np.array(ssim_mean) + np.array(ssim_sem),
                    color='darkblue',
                    alpha=0.5)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=log_x[np.argmax(ssim_mean)],
              ymin=ymin,
              ymax=ssim_mean[np.argmax(ssim_mean)],
              colors='k',
              linestyles='dashed')
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(log_x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('\nWeighting Coefficient \u03BB (Log Scale)', fontsize=12)
    ax.set_ylabel('SSIM\n(HIGHER is better)', fontsize=12)

    ax = fig.add_subplot(1, 4, 3)
    ax.plot(log_x, ergas_mean, c='firebrick', linewidth=2)
    ax.fill_between(log_x,
                    np.array(ergas_mean) - np.array(ergas_sem),
                    np.array(ergas_mean) + np.array(ergas_sem),
                    color='firebrick',
                    alpha=0.5)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=log_x[np.argmin(ergas_mean)],
              ymin=ymin,
              ymax=ergas_mean[np.argmin(ergas_mean)],
              colors='k',
              linestyles='dashed')
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(log_x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('\nWeighting Coefficient \u03BB (Log Scale)', fontsize=12)
    ax.set_ylabel('ERGAS\n(LOWER is better)', fontsize=12)

    ax = fig.add_subplot(1, 4, 4)
    ax.plot(log_x, rmse_mean, c='firebrick', linewidth=2)
    ax.fill_between(log_x,
                    np.array(rmse_mean) - np.array(rmse_sem),
                    np.array(rmse_mean) + np.array(rmse_sem),
                    color='firebrick',
                    alpha=0.5)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=log_x[np.argmin(rmse_mean)],
              ymin=ymin,
              ymax=rmse_mean[np.argmin(rmse_mean)],
              colors='k',
              linestyles='dashed')
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(log_x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('\nWeighting Coefficient \u03BB (Log Scale)', fontsize=12)
    ax.set_ylabel('RMSE\n(LOWER is better)', fontsize=12)

    plt.tight_layout()
    plt.savefig('../../results/lambda_test.png')