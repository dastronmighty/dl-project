import os

import pandas as pd
import numpy as np

from matplotlib import pylab as plt


def rename_cols(df_):
    """
    renames the Wrongly named "Test" in the logs to "Validation"
    :param df_: the df to rename
    :return: dataframe with renamed columns
    """
    cols = df_.columns
    col_dict = {}
    for c in cols:
        if "Test" in c:
            col_dict[c] = c.replace("Test", "Validation")
        else:
            col_dict[c] = c
    return df_.rename(columns=col_dict)


def log_to_df(file_name, test_str=""):
    """
    Transform a log file to a dataframe
    :param file_name: the name of the file (its actually the full path)
    :param test_str: a variable you are able to pass in to test each log line whether it should be parsed or not if the test string is in it
    :return: the name of the model and the Dataframe
    """
    text = ""
    name = file_name.split("/")[-1].replace(".txt", "")
    with open(file_name) as f:
        text = str(f.read())
    lines = text.split("\n")
    log_arr_dict = []
    for l in lines:
        if test_str in l.lower():
            g = {}
            for f in l.split("-"):
                if ":" in f:
                    fn = f.split(":")[0].strip()
                    fv = float(f.split(":")[1].strip())
                    g[fn] = fv
            log_arr_dict.append(g)
    return name, pd.DataFrame(log_arr_dict)


def get_logs_from_dir_helper(dir_name):
    """
    A recursive funciton that gets all the text file paths from a directory (log directory)
    :param dir_name: "the path to the log directory to process"
    :return: all the text files form the log directory as a list
    """
    logs = []
    for _ in os.listdir(dir_name):
        p = f"{dir_name}/{_}"
        if os.path.isdir(p):
            u_ps = get_logs_from_dir_helper(p)
            logs += u_ps
        elif ".txt" in p:
            logs.append({"path": p, "filename": _})
    return logs


def get_logs_from_dir(dir_name):
    """
    get all the logs from a directory
    :param dir_name: the directory path
    :return: the logs and the final logs
    """
    logs = get_logs_from_dir_helper(dir_name)
    names = {}
    for log in logs:
        if log['filename'] not in names.keys():
            names[log['filename']] = log['path']
    logs = list(names.values())
    finals = [l for l in logs if "final" in l.lower()]
    logs = [l for l in logs if "final" not in l.lower()]
    return logs, finals


def finals_to_df(logs):
    """
    Process all the final tests of a model into a dataframe
    :param logs: the final logs for a model
    :return: the dataframe
    """
    n, df = log_to_df(logs[0], "final")
    df["name"] = n
    for _ in logs[1:]:
        n, df1 = log_to_df(_, "final")
        df1["name"] = n
        df = df.append(df1)
    return df


def dfs_from_logs(dir_name):
    """
    get all the dataframes associated with a log directory
    :param dir_name: the path to the log directory
    :return: the name of the model, the logs as dfs in a dict for each hyperparam and the final test dataframe
    """
    logs, final_logs = get_logs_from_dir(dir_name)
    all_logs = {}
    for l in logs:
        n, df = log_to_df(l, "epoch")
        if len(df) > 0:
            df = rename_cols(df)
            all_logs[n] = df
    finals_df = finals_to_df(final_logs)
    name = (dir_name.split("/")[-1]
    return name, all_logs, finals_df


def gen_feature_plot(name,
                     log_df,
                     feats,
                     figsize=(15, 15),
                     main_font_size=20):
    """
    Generate a plot from a given list of fetaures for a dataframe
    :param name: the name of the plot
    :param log_df: the dataframe to use
    :param feats: the features to plot
    :param figsize: the size
    :param main_font_size: the font size
    :return: the figure returned by matplot lib
    """
    epochs = [J for J in range(len(log_df))]
    f, axs = plt.subplots(1, 1, figsize=figsize)
    for fe in feats:
        feat = log_df[fe]
        axs.plot(epochs, feat)
    axs.legend(feats)
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Score")
    f.suptitle(f"{name} - {(' vs ').join(feats)}", fontsize=main_font_size)
    return f


def gen_feature_collage(logs,
                        feats,
                        figsize=(15, 15),
                        smol_font_size=8,
                        save_to=".",
                        main_font_size=30):
    """
    Generate a plot for all dataframes in a dictionary into a collage from a given list of fetaures for a dataframe
    :param logs: the log dictionary to use
    :param feats: the features to plot
    :param figsize: the size
    :param save_to: where to save the plot
    :param main_font_size: the font size
    :return: the figure returned by matplot lib
    """
    if len(logs.keys()) > 1:
        s = int(np.ceil(np.sqrt(len(logs))))
        f, axs = plt.subplots(s, s, figsize=figsize)
        for i, log in enumerate(logs.keys()):
            epochs = [J for J in range(len(logs[log]))]
            for fe in feats:
                feat = logs[log][fe]
                axs[i // s, i % s].plot(epochs, feat)
            axs[i // s, i % s].legend(feats)
            axs[i // s, i % s].set_title(f"{log} - {(' vs ').join(feats)}", fontsize=smol_font_size)
            axs[i // s, i % s].set_xlabel('Epoch', fontsize=smol_font_size)
            axs[i // s, i % s].set_ylabel("Score", fontsize=smol_font_size)
        for i in range(len(list(logs.keys())), (s * s)):
            axs[i // s, i % s].axis('off')
        title_name = list(logs.keys())[0].split("_")[0]
        f.suptitle(f"{title_name} - {(' vs ').join(feats)}", fontsize=main_font_size)
        f.savefig(f"{save_to}/collage_{title_name}_{('_vs_').join(feats)}")
        return f


def gen_feature_pics(log_dict, feats, save_to=".", **kwargs):
    """
    Generate a plot for each dataframe in a dictionary  from a given list of fetaures for a dataframe
    :param log_dict: the log dictionary to use
    :param feats: the features to plot
    :param save_to: where to save the plot
    :param kwargs: other arguments to pass to the gen_feature_plot
    """
    names = list(log_dict.keys())
    for n in names:
        f = gen_feature_plot(n, log_dict[n], feats, **kwargs)
        f.savefig(f"{save_to}/single_{n}_{('_vs_').join(feats)}")


def autolabel(rects, axs, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for rect in rects:
        height = rect.get_height()
        axs.annotate('{}'.format(round(height, 3)),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                     textcoords="offset points",  # in both directions
                     ha=ha[xpos], va='bottom')


def gen_final_hist(name,
                   final_df,
                   figsize=(15, 15),
                   main_font_size=16,
                   save_to=".",
                   width=0.3):
    """
    Generate a Histogram from a final dataframe to show the Test accuracy and Test AUC of each of a models different hyperparams
    :param name: the name to use
    :param final_df: the final dataframe to use
    :param figsize: the fig size
    :param main_font_size: the font size
    :param save_to: where to save the plot
    :param width: the width of the bars
    :return: the figure returned by matplotlib
    """
    feats = ["Test auc", "Test acc"]
    names = final_df["name"].apply(lambda x: '_'.join(x.split('_')[2:5]))
    f, axs = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(final_df))
    r1 = axs.bar((x - width / 2), final_df[feats[0]], width)
    r2 = axs.bar((x + width / 2), final_df[feats[1]], width)
    autolabel(r1, axs, "left")
    autolabel(r2, axs, "right")
    axs.set_xticks(x)
    axs.set_xticklabels(names)
    axs.set_xlabel('learning rate _ batch size', fontsize=main_font_size)
    axs.set_ylabel("Score", fontsize=main_font_size)
    axs.legend(feats)
    f.suptitle(f"{name} - Final {' vs '.join(feats)} Results", fontsize=main_font_size)
    f.savefig(f"{save_to}/FINAL_{name}_{('_vs_').join(feats)}")
    return f


def get_log_dirs(p):
    """
    get all the subdirectories with 'log' in their name from a given directory
    :param p: the path to get the dirs from
    :return: the list of log directorys
    """
    return [f"{p}/{_}" for _ in os.listdir(p) if "log" in _.lower()]
