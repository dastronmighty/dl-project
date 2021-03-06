import os
import math

import torch
import torchvision

import numpy as np

import pandas as pd

from tqdm import tqdm

class Preproccessor:
    def __init__(self, data_path, output_path, from_, to_, MAX_SIZE=512, verbose=False):
        if data_path is None:
            raise RuntimeError("No path to data directory given")
        data_dir = f"{data_path}/ODIR-5K/ODIR-5K/Training Images"
        self.verbose = verbose
        self.max_size = MAX_SIZE

        self.resizer = torchvision.transforms.Resize(size=(MAX_SIZE, MAX_SIZE))
        self.to_img = torchvision.transforms.ToPILImage()

        data_df = pd.read_csv(f"{data_path}/full_df.csv")[from_:to_]
        data_df["path"] = data_df["filename"].apply(lambda x: f"{data_dir}/{x}")
        data_df = data_df[["N","path"]]

        base = f"{output_path}/preprocessed_data_images"

        if not os.path.isdir(base):
            os.mkdir(base)
        else:
            raise RuntimeError(f"{base} already exists")

        self.decide_output("Pre-Processing and saving Data")
        self.load_from_df(data_df,base)



    def decide_output(self, msg):
        if self.verbose:
            print("\n"+msg)

    def load_from_df(self, df, n):
        """
        Load data from df

        Parameters
        ----------
        df : dataframe
            the dataframe to load in data from
        """
        labels = self.get_labels(df)
        self.load_and_process_arr(df["path"], labels, n)


    def get_labels(self, df):
        labels = np.array(df["N"].values, dtype=np.uint8)
        return labels

    def load_and_process_arr(self, arr, labels, savedir):
        """
        Load images and process them from files

        Parameters
        ----------
        arr : list
            list of paths to images

        Returns
        -------
        array : numpy array
            numpy array of loaded, transofrmed and flattened images
        """
        i = 0
        for p in tqdm(arr, disable=(not self.verbose)):
            img = torchvision.io.read_image(p)
            img = self.transform_image(img)
            self.to_img(img).save(f"{savedir}/{i:04}_l{labels[i]}.jpg")
            i += 1

    def transform_image(self, img):
        """
        Transform a given input image

        Parameters
        ----------
        img : tensor
            the image to transform

        Returns
        -------
        img : tensor
            transformed image
        """
        n_img = self.remove_border(img)
        n_img = self.resizer(n_img)
        n_img = np.array(n_img)
        n_img = torch.tensor(n_img, dtype=torch.uint8)
        return n_img

    def remove_bord_helper(self, img, to_idx, check_against, row):
        """
        find the start and ending locations of the non-blank parts of the image along the rows/columns

        Parameters
        ----------
        img : tensor
            the image to crop
        to_idx : int
            the maximum index of the image to check
        check_against : tensor
            a 1xN slice to check against the slices of the input image to determine the start and end positions
            if the beginning/ending slices of the image match the check against they are removed
        row: boolean
            wether we are processing a row or a column

        Returns
        -------
        list
            indexes of the crop locations along the axis we chose (row/columns)
        """
        check_ = lambda x: torch.equal(img[:,x], check_against) if row else torch.equal(img[:,:,x], check_against)
        i_1, i_2 = 0, 0
        for i in range(to_idx):
            if i_1 == 0: # if we are haven't reached image yet
                if not check_(i): # if the slice is not just a black bar on the left/top side
                    i_1 = i
            if i_1 != 0 and i_2 == 0: # if we havent gone passed the image yet
                if check_(i): # if the slice is just a black bar on the right/bottom side
                    i_2 = i
        if i_1 > (check_against.shape[1]/2):
            i_1 = 0
        if i_2 < (check_against.shape[1]/2):
            i_2 = to_idx
        return i_1, i_2

    def remove_border(self, img):
        """
        remove black borders of an image

        Parameters
        ----------
        img : tensor
            the image to crop

        Returns
        -------
        img : tensor
            cropped image
        """
        z_row = torch.tensor(np.zeros(img[:,0].shape), dtype=torch.uint8) # a black row slice of the input image
        z_col = torch.tensor(np.zeros(img[:,:,0].shape), dtype=torch.uint8) # a black column slice of the input image
        i_t_row, i_b_row = self.remove_bord_helper(img, img.shape[1], z_row, True) # the start and end rows of the image
        i_l_col, i_r_col = self.remove_bord_helper(img, img.shape[2], z_col, False) # the start and end columns of the image
        return img[:,i_t_row:i_b_row,i_l_col:i_r_col] # return the cropped image
