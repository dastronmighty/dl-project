import torchvision
import os
from tqdm import tqdm


class OcularAugmenter:
    '''
    Ocular Augmenter


    '''

    def __init__(self,
                 DATA_DIR,
                 OUT_DIR,
                 size_out,
                 rotation_angle=45,
                 invert=False,
                 equalize=False,
                 autocontrast=False,
                 verbose=False):
        """
        Creating a new Ocular Augmenter Class
        :param DATA_DIR: Data directory
        :param OUT_DIR: the output directory
        :param size_out: the size of the output augmented images
        :param rotation_angle: the angle to rotate the images by
        :param invert: whether or not to also invert the colors
        :param equalize: whether or not to also equalize the colors
        :param autocontrast: whether or not to also apply autocontrast to the colors
        :param verbose: the verbosity of the operations
        """
        self.out_dir = f"{OUT_DIR}/augmented"
        os.makedirs(f"{self.out_dir}")
        self.angle = rotation_angle
        self.verbose = verbose

        self.files = [f"{DATA_DIR}/{_}" for _ in os.listdir(DATA_DIR)]

        self.resizer = torchvision.transforms.Resize(size=(size_out, size_out))
        self.to_img = torchvision.transforms.ToPILImage()

        self.img_number = 0

        os.makedirs(f"{self.out_dir}/resized")

        os.makedirs(f"{self.out_dir}/rotated")

        self.transforms = []
        if invert:
            self.transforms += [
                {"name": "invert",
                 "function": torchvision.transforms.functional.invert}]
            os.makedirs(f"{self.out_dir}/invert")
        if equalize:
            self.transforms += [
                {"name": "equalize",
                 "function": torchvision.transforms.functional.equalize}]
            os.makedirs(f"{self.out_dir}/equalize")
        if autocontrast:
            self.transforms += [
                {"name": "autocontrast",
                 "function": torchvision.transforms.functional.autocontrast}]
            os.makedirs(f"{self.out_dir}/autocontrast")

        for p in tqdm(self.files, disable=(not self.verbose)):
            self.augment(p)

    def augment(self, file_):
        """
        The main augment loop for an image
        :param file_: the file of the image to augment
        """
        label = file_[-5:-4]
        img = torchvision.io.read_image(file_)
        n_img = self.resizer(img)
        self.save(n_img, label, f"resized")
        for i in range(0, 360, self.angle): # go through angles
            rotated = torchvision.transforms.functional.rotate(n_img, angle=i)
            self.save(rotated, label, f"rotated")
            for transform in self.transforms: # apply each transform that we requested
                t_img = transform["function"](rotated)
                self.save(t_img, label, f"{transform['name']}")

    def save(self, img, label, tag):
        """
        save the image
        :param img: the tensor to save as an image
        :param label: the lable of the image
        :param tag: a tag for the image to designate what augmentations have happened to it
        :return:
        """
        name = f"{str(self.img_number).zfill(6)}_{label}"
        self.to_img(img).save(f"{self.out_dir}/{tag}/{name}.jpg")
        self.img_number += 1