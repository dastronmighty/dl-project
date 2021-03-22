from src.utils.Plotting import get_log_dirs, dfs_from_logs, gen_feature_pics, gen_feature_collage, gen_final_hist
from src.utils.utils import init_folder

LOG_DIR = "./logs"
FIG_DIR = "./figs"


def main():
    log_directories = get_log_dirs(LOG_DIR)
    features_to_plot = [
        ["Train Loss", "Validation Loss"],
        ["Train auc", "Validation auc"],
        ["Train acc", "Validation acc"],
        ["Train Loss", "Validation Loss",
         "Train auc", "Validation auc",
         "Train acc", "Validation acc"]
    ]
    for log_dir in log_directories:
        name, all_logs, finals_df = dfs_from_logs(log_dir)
        save_dir = init_folder(name, FIG_DIR, True)
        for feats_to_plt in features_to_plot:
            f = gen_feature_pics(all_logs, feats_to_plt, save_to=save_dir)
            f = gen_feature_collage(all_logs, feats_to_plt, save_to=save_dir, smol_font_size=10, figsize=(25, 25))
            f = gen_final_hist(name, finals_df, save_to=save_dir)


if __name__ == '__main__':
    main()




