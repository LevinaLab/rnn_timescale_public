import torch
import torch.nn.functional as F
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(M) < torch.rand(1)) * 1.

# Experiments (not included in paper) with sparse sequences
def generate_sparse_binary_sequence(M, sparsity=0.9):
    s = torch.rand(M) * 2 - 1
    s = torch.where(torch.abs(s) > sparsity, torch.sign(s), 0 * s)
    return s * 1.



############ N_PARITY TASKS ##############

def get_parity(vec, N):
    return (vec[-N:].sum() % 2).long()

def make_batch_Nbit_pair_parity(Ns, bs, classify_in_time=False):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
        if classify_in_time:
            labels = [torch.stack([get_parity_in_time(s, N) for s in sequences]) for N in Ns]
        else:
            labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]

        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)
    return sequences, labels


def get_parity_in_time(vec, N):
    labels = []
    for idx in np.arange(N, len(vec)):
        vec_t = vec[idx-N:idx]
        l = get_parity(vec_t, N)
        labels.append(l)
        # labels.append(((vec_t + 1) / 2)[-N:].sum() % 2)

    labels = torch.stack(labels).long()

    return labels


############ DMS TASKS ##################

def get_match(vec, N):
    return (vec[-N] == vec[-1]).long()

def make_batch_multihead_dms(Ns, bs):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M, balanced=True).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_match(s, N) for s in sequences]).squeeze() for N in Ns]

        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)

    return sequences, labels


### Pulsed Parity Task ###

def generate_pulsed_binary_sequence(M, pulse_time):
    # generate a binary signal [-1, 1] that is seperated by a pusle time.
    # this signal is used to be convolved with the gaussian kernel
    s = torch.sign(torch.rand(M) * 2 - 1)
    s = torch.stack([s_i if i % pulse_time == 0 and i > 0 else 0 * s_i for i, s_i in enumerate(s)])
    # s = torch.where(torch.abs(s) > sparsity, torch.sign(s), 0 * s)
    return s * 1.


def gaussian_kernel(size, sigma=1):
    size = 2 * size + 1
    """Generates a Gaussian kernel."""
    kernel = torch.from_numpy(np.fromfunction(
        lambda x: (1 / (2 * np.pi * sigma ** 2)) * np.exp(- (x - size // 2) ** 2 / (2 * sigma ** 2)),
        (size,)
    ))
    return kernel / kernel.sum()


def smooth_sequence(input_sequence, kernel_size, sigma):
    """Smooths a 1D sequence using a Gaussian kernel."""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, -1).float()  # Reshape for conv1d
    smoothed_sequence = F.conv1d(input_sequence.view(1, 1, -1).float(), kernel, padding=kernel_size // 2)
    return smoothed_sequence.squeeze() / smoothed_sequence.abs().max()


def make_batch_Nbit_pair_parity_pulsed(Ns, bs, pulse_time=10, kernel_size=3, sigma=1):
    M_min = pulse_time * Ns[-1] + pulse_time
    M_max = M_min + 3 * Ns[-1] * pulse_time
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_pulsed_binary_sequence(M, pulse_time=pulse_time).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_parity(s[::pulse_time][1:] / 2 + 0.5, N) for s in sequences]) for N in Ns]

        # sequences = torch.stack(sequences)
        # sequences = sequences.permute(1, 0, 2)

        sequences = torch.stack(sequences).permute(0, 2, 1)
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, -1).float()  # Reshape for conv1d
        sequences = F.conv1d(sequences, kernel, padding=kernel_size // 2)
        sequences = sequences / sequences.abs().max(dim=-1, keepdim=True)[0]
        sequences = sequences.permute(2, 0, 1)
    return sequences, labels

'''Commented out for slurm reasons
import torch.nn.functional as F
from torchvision import datasets, transforms

############ sMNIST TASKS ##################

# Load MNIST dataset


def init_mnist():
    # load the training and sets for MNIST
    # (you must download the data once and put it in the right directory manually)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # transform = transforms.Compose([
    #     transforms.ToTensor()])


    train_set = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./datasets', train=False, download=True, transform=transform)

    return train_set, test_set

def rescale(images, new_L):
    """
    Rescale the images

    Parameters:
        images (torch.Tensor): A tensor containing images with shape (batch_size, channels, L, L).
        new_L (int): The new width to which the images should be rescaled.

    Returns:
        torch.Tensor: A tensor containing the rescaled images with shape (batch_size, channels, new_L, new_L).
    """

    # Perform interpolation for rescaling
    rescaled_images = F.interpolate(images, size=(new_L, new_L), mode='bilinear')

    return rescaled_images
def get_smnist_batch(Ns, bs, data):
    # Choose 'bs' random images from MNIST dataset
    indices = np.random.randint(0, len(data), bs)


    ######## MULTIHEAD IMPLEMENTATION ######## (slow and sucks)
    # batch_images = [data[i][0] for i in indices]
    # # random rescalings (for multi-head, this uses the batch dimension to train multiple scales simultaneously)
    # rs = np.random.choice(Ns, size=(256,)) + 3
    #
    # r_data = [rescale(batch_images[i], new_L=rs[i]).reshape(1, -1) for i in range(bs)]
    # s_lengths = [d.shape[1] for d in r_data]
    # max_length = np.max(s_lengths)
    # # random padding
    # M = np.random.randint(M_min, M_max)
    # # pad to max sequence length + random padding
    # pad_s = max_length - np.array(s_lengths) + M
    # M1 = (np.random.rand(bs) * pad_s).astype('int')
    # M2 = pad_s - M1
    # r_data = torch.stack([F.pad(r_data[i], (M1[i], M2[i])) for i in range(bs)])
    # sequences = r_data.permute(2, 0, 1)
    ########################################

    ######## SINGLEHEAD IMPLEMENTATION ########
    batch_images = torch.stack([data[i][0] for i in indices])
    r_data = rescale(batch_images, new_L=Ns[-1] + 3)

    M_min = 2
    M_max = M_min + Ns[-1]
    M = np.random.randint(M_min, M_max)
    # padding values
    M1 = np.random.randint(0, M)
    M2 = M - M1

    r_data = F.pad(r_data, (0, 0, M1, M2))
    sequences = r_data.reshape(bs, -1, 1).permute(1, 0, 2)
    ########################################

    # this list of labels is redundant, but needs to be done to work with the multi-head version
    batch_labels = [torch.tensor([data[i][1] for i in indices], dtype=torch.long)]

    return sequences, batch_labels
    
'''