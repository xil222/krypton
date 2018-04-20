import torch
import h5py


def load_dict_from_hdf5(filename, cuda=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', cuda)


def __recursively_load_dict_contents_from_group(h5file, path, cuda=True):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = torch.from_numpy(item.value)
            if cuda:
                ans[key] = ans[key].cuda()

        elif isinstance(item, h5py._hl.group.Group, cuda):
            ans[key] = __recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
