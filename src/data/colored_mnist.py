import numpy as np
import torch as T
from torchvision import transforms
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.utils as vutils
from src import ROOT_PATH
from typing import Tuple
from tqdm import tqdm
import logging
from collections.abc import Iterable


logger = logging.getLogger(__name__)


class SCMDataTypes:
    BINARY = "binary"
    REP_BINARY = "rep_binary"
    BINARY_ONES = "binary_ones"
    REP_BINARY_ONES = "rep_binary_ones"
    ONE_HOT = "one_hot"
    REAL = "real"
    IMAGE = "image"


class SCMDataGenerator:
    def __init__(self, mode="sampling", normalize=True):
        self.v_size = {}
        self.v_type = {}
        self.cg = None
        self.mode = mode
        self.normalize = normalize

    def generate_samples(self, n):
        return None

def check_equal(input, val):
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)

def show_image_grid(batch, dim_treat, dir=None, title=None):
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.title(title)
    grid = vutils.make_grid(batch[: 64], padding=2, normalize=True, nrow=dim_treat).cpu()
    # grid = vutils.make_grid(batch.to(device)[: 64], padding=2).cpu()
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    if dir is not None:
        plt.savefig(dir)
    else:
        plt.show()
    plt.close()


def expand_do(val, n):
    return np.ones(n, dtype=int) * val


class ColorMNISTDataGenerator(SCMDataGenerator):
    def __init__(self, image_size, mode, split, evaluating=False):
        super().__init__(mode)

        self.evaluating = evaluating
        self.raw_mnist_n = 0
        self.raw_mnist_images = None
        if not evaluating:
            mnist_data = MNIST(f'{ROOT_PATH}/data/hcmnist/MNIST/raw')
            if split == 'train':
                images, labels = mnist_data.load_training()
            else:
                images, labels = mnist_data.load_testing()
            self.raw_mnist_n = len(images)
            images = np.array(images).reshape((self.raw_mnist_n, 28, 28))
            labels = np.array(labels)

            self.raw_mnist_images = dict()
            for i in range(len(labels)):
                if labels[i] not in self.raw_mnist_images:
                    self.raw_mnist_images[labels[i]] = []
                self.raw_mnist_images[labels[i]].append(images[i])

        self.colors = {
            0: (1.0, 0.0, 0.0),
            1: (1.0, 0.6, 0.0),
            2: (0.8, 1.0, 0.0),
            3: (0.2, 1.0, 0.0),
            4: (0.0, 1.0, 0.4),
            5: (0.0, 1.0, 1.0),
            6: (0.0, 0.4, 1.0),
            7: (0.2, 0.0, 1.0),
            8: (0.8, 0.0, 1.0),
            9: (1.0, 0.0, 0.6)
        }

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor()
        ])

        self.mode = mode
        if mode == "sampling_noncausal":
            self.v_size = {
                'digit': 10,
                'image': 3
            }
            self.v_type = {
                'digit': SCMDataTypes.ONE_HOT,
                'image': SCMDataTypes.IMAGE
            }
            self.cg = "color_mnist_noncausal"
        else:
            self.v_size = {
                'color': 10,
                'digit': 10,
                'image': 3
            }
            self.v_type = {
                'color': SCMDataTypes.ONE_HOT,
                'digit': SCMDataTypes.ONE_HOT,
                'image': SCMDataTypes.IMAGE
            }
            self.cg = "color_mnist"

    def colorize_image(self, image, color):
        color_value = self.colors[color]
        h, w = image.shape
        new_image = np.reshape(image, [h, w, 1])
        new_image = np.concatenate([new_image * color_value[0], new_image * color_value[1], new_image * color_value[2]],
                                   axis=2)
        return new_image

    def sample_digit(self, digit, color):
        total = len(self.raw_mnist_images[digit])
        ind = np.random.randint(0, total)
        img_choice = np.round(self.colorize_image(self.raw_mnist_images[digit][ind], color)).astype(np.uint8)

        return img_choice

    def generate_samples(self, n, treat_dim=2, U={}, do={}, p_align=0.85, return_U=False, normalize=True, colors=None):
        if "u_conf" in U:
            u_conf = U["u_conf"]
        else:
            u_conf = np.random.randint(treat_dim, size=n)
        if "u_digit" in U:
            u_digit = U["u_digit"]
        else:
            u_digit = np.random.randint(treat_dim, size=n)
        if "u_color" in U:
            u_color = U["u_color"]
        else:
            u_color = np.random.randint(treat_dim, size=n)
        if "u_dig_align" in U:
            u_dig_align = U["u_dig_align"]
        else:
            u_dig_align = np.random.binomial(1, p_align, size=n)
        if "u_color_align" in U:
            u_color_align = U["u_color_align"]
        else:
            u_color_align = np.random.binomial(1, p_align, size=n)

        if "digit" in do:
            digit = do["digit"]
        else:
            digit = np.where(u_dig_align, u_conf, u_digit)

        if "color" in do:
            color = do["color"]
        else:
            if colors is None:
                color = np.where(u_color_align, u_conf, u_color)
            else:
                color = colors


        one_hot_digits = np.zeros((n, treat_dim))
        one_hot_digits[np.arange(n), digit] = 1
        one_hot_colors = np.zeros((n, treat_dim))
        one_hot_colors[np.arange(n), color] = 1

        imgs_f = []
        imgs_pot = [[] for _ in range(treat_dim)]
        logger.info(f'Sampling {n} colored digits.')
        for i in tqdm(range(n)):
            img_sample = self.sample_digit(digit[i], color[i])
            img_sample = self.transform(img_sample).float()
            if normalize:
                img_sample = 2.0 * img_sample - 1.0
            else:
                img_sample = 255.0 * img_sample
            imgs_f.append(img_sample)

            for pot_digit in range(treat_dim):
                img_sample = self.sample_digit(pot_digit, color[i])
                img_sample = self.transform(img_sample).float()
                if normalize:
                    img_sample = 2.0 * img_sample - 1.0
                else:
                    img_sample = 255.0 * img_sample
                imgs_pot[pot_digit].append(img_sample)

        if self.mode == "sampling_noncausal":
            data = {
                'digit': T.tensor(one_hot_digits).float(),
                'image': T.stack(imgs_f, dim=0)
            }
        else:
            data = {
                'color': T.tensor(one_hot_colors).float(),
                'digit': T.tensor(one_hot_digits).float(),
                'image': T.stack(imgs_f, dim=0),
                'image_pot': [T.stack(imgs, dim=0) for imgs in imgs_pot]
            }

        if return_U:
            new_U = {
                "u_conf": u_conf,
                "u_digit": u_digit,
                "u_color": u_color,
                "u_dig_align": u_dig_align,
                "u_color_align": u_color_align
            }
            return data, new_U
        return data

    def sample_pot(self, digit, colors, normalize=True):
        imgs_pot = []

        for i in range(len(colors)):
            img_sample = self.sample_digit(digit, colors[i])
            img_sample = self.transform(img_sample).float()
            if normalize:
                img_sample = 2.0 * img_sample - 1.0
            else:
                img_sample = 255.0 * img_sample
            imgs_pot.append(img_sample)

        return T.stack(imgs_pot, dim=0)



    def sample_ctf(self, q, n=64, batch=None, max_iters=1000, p_align=0.85, normalize=True):
        if batch is None:
            batch = n

        iters = 0
        n_samps = 0
        samples = dict()

        while n_samps < n:
            if iters >= max_iters:
                return float('nan')

            new_samples = self._sample_ctf(batch, q, p_align=p_align, normalize=normalize)
            if isinstance(new_samples, dict):
                if len(samples) == 0:
                    samples = new_samples
                else:
                    for var in new_samples:
                        samples[var] = T.concat((samples[var], new_samples[var]), dim=0)
                        n_samps = len(samples[var])

            iters += 1

        return {var: samples[var][:n] for var in samples}

    def _sample_ctf(self, n, q, p_align=0.85, normalize=True):
        _, U = self.generate_samples(n, return_U=True, p_align=p_align, normalize=normalize)

        n_new = n
        for term in q.cond_term_set:
            samples = self.generate_samples(n=n_new, U=U, do={
                k: expand_do(v, n_new) for (k, v) in term.do_vals.items()
            }, return_U=False, p_align=p_align, normalize=normalize)

            cond_match = T.ones(n_new, dtype=T.bool)
            for (k, v) in term.var_vals.items():
                cond_match *= check_equal(samples[k], v)

            U = {k: v[cond_match] for (k, v) in U.items()}
            n_new = T.sum(cond_match.long()).item()

        if n_new <= 0:
                return float('nan')

        out_samples = dict()
        for term in q.term_set:
            expanded_do_terms = dict()
            for (k, v) in term.do_vals.items():
                    expanded_do_terms[k] = expand_do(v, n_new)
            q_samples = self.generate_samples(n=n_new, U=U, do=expanded_do_terms, return_U=False, p_align=p_align,
                                              normalize=normalize)
            out_samples.update(q_samples)

        return out_samples

    def show_image(self, image, label=None, dir=None):
        if label is not None:
            plt.title('Label is {label}'.format(label=label))
        image = T.movedim(image, 0, -1)
        image = (image + 1.0) / 2.0
        plt.imshow(image)

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.clf()

    def show_legend(self, dir=None, title="Legend"):
        photos = []
        for i in range(10):
            digit = self.sample_digit(i, i)
            digit = self.transform(digit).float()
            digit = 2.0 * digit - 1.0
            photos.append(digit)
        photos = T.stack(photos, dim=0)
        plt.figure(figsize=(10, 1))
        plt.axis("off")
        plt.title(title)
        grid = vutils.make_grid(photos, padding=2, normalize=True, nrow=10).cpu()
        plt.imshow(np.transpose(grid, (1, 2, 0)))

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.close()

    def show_gradient(self, dir=None):
        def color_mix(p, phase):
            if phase == 0:
                return (1.0, p, 0.0)
            elif phase == 1:
                return (1.0 - p, 1.0, 0.0)
            elif phase == 2:
                return (0.0, 1.0, p)
            elif phase == 3:
                return (0.0, 1.0 - p, 1.0)
            elif phase == 4:
                return (p, 0.0, 1.0)
            elif phase == 5:
                return (1.0, 0.0, 1.0 - p)
            else:
                return (0.0, 0.0, 0.0)

        n = 600
        fig, ax = plt.subplots(figsize=(8, 2))
        for x in range(n):
            phase = x // 100
            p = (x % 100) / 100.0
            color = color_mix(p, phase)
            ax.axvline(x, color=color, linewidth=4)
        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.close()


class ConditionedColoredMNIST:

    def __init__(self, digit, cov_f, subset):
        self.digit = digit
        self.cov_f = cov_f
        self.subset = subset

    def sample(self, n_samples):
        imgs = []
        for i in tqdm(range(n_samples[0])):
            colors = np.where(self.cov_f == 1.0)[1]
            img = self.subset.sample_pot(self.digit, colors)
            img = img.reshape((img.shape[0], -1))
            imgs.append(T.tensor(img))
        return T.stack(imgs, dim=0)


class ColoredMNIST:

    def __init__(self, dim_treat=10, img_size=10, n_train=60000, n_test=10, conf_p=0.5, **kwargs):
        self.train_subset = ColorMNISTDataGenerator(img_size, "sampling", split='train')
        self.test_subset = ColorMNISTDataGenerator(img_size, "sampling", split='test')
        self.dim_treat = dim_treat
        self.n_train = n_train
        self.n_test = n_test
        self.conf_p = conf_p

    def get_data(self):
        train_data = self.train_subset.generate_samples(self.n_train, treat_dim=self.dim_treat, p_align=self.conf_p)
        test_data = self.train_subset.generate_samples(self.n_test, treat_dim=self.dim_treat, p_align=self.conf_p,
                                                       colors=np.concatenate([np.arange(self.dim_treat)] * (self.n_test // self.dim_treat), axis=0))

        data_dicts = []

        for data, subset_name in zip([train_data, test_data], ['train', 'test']):
            data_dict = {
                'cov_f': data['color'].cpu().numpy(),
                'treat_f': data['digit'].cpu().numpy(),
                'out_f': data['image'].reshape((data['image'].shape[0], -1)).cpu().numpy(),
                'out_f_scaled': data['image'].reshape((data['image'].shape[0], -1)).cpu().numpy(),
                'subset': subset_name,
            }

            for treat_pot in range(self.dim_treat):
                data_dict[f'out_pot{treat_pot}'] = data['image_pot'][treat_pot].reshape((data['image_pot'][treat_pot].shape[0], -1)).cpu().numpy()
                data_dict[f'out_pot{treat_pot}_scaled'] = data_dict[f'out_pot{treat_pot}']

            data_dicts.append(data_dict)

        return data_dicts

    def get_pot_cond_dist(self, data_dict) -> Tuple[ConditionedColoredMNIST]:
        cov_f = data_dict['cov_f']
        subset = self.train_subset if data_dict['subset'] == 'train' else self.test_subset
        return tuple([ConditionedColoredMNIST(treat, cov_f, subset) for treat in range(self.dim_treat)])



if __name__ == "__main__":
    mdg = ColorMNISTDataGenerator(10, "sampling", split='train')

    data = mdg.generate_samples(10)
    print(data['color'])
    print(data['digit'])
    for i in range(len(data['image'])):
        mdg.show_image(data['image'][i])

    mdg.show_legend("legend.png")