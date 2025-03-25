import matplotlib.pyplot as plt
import pandas as pd

def read_logs(filename):

    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":

    logs_file = "clone/C0_P0/log.txt"
    log_df = read_logs(logs_file)

    gen = log_df['Generation']
    fit_mean = log_df['Mean']
    fit_max = log_df['Max']
    fit_min = log_df['Min']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # ax.plot(gen, fit_mean, label ='mean fitness')
    ax.plot(gen, fit_max, label ='max fitness')
    # ax.plot(gen, fit_min, label ='min fitness')
    ax.grid(True)
    ax.set_xlabel("generation")
    ax.set_ylabel("Action Agreement Ratio")
    ax.legend()
    plt.show()
    
