import torchnet as tnt
from torch.utils.data.dataloader import default_collate


class dataloader(object):
    def __init__(self,
                 idx=None,
                 dataset=None,
                 batch_size=None,
                 epochs = None,
                 num_workers=None,
                 shuffle=None,
                 transformation=None,
                 representation=None,
                 name=None,
                 description=None,
                 test=False):

        self.idx = idx
        self.dataset = dataset
        self.name = name
        self.size = None
        self.n_samples = None
        self.image = 0
        self.image_vector = 0
        self.vector = 0
        self.graph = 0
        self.description = description
        self.test = test
        self.epoch_size = epochs if epochs is not None else len(dataset)

        if representation == 'image':
            self.image = 1
            self.size = 0#self.dataset.x[0]['image'].shape[1:](TODO)
            self.n_samples = 0#len(self.dataset.x)(TODO)
        elif representation == 'imageVector':
            self.image_vector = 1
        elif representation == 'vector':
            self.vector = 1
        elif representation == 'graph':
            self.graph = 1
        else:
            print(
                'Please select the representation as either: Image, ImageVector, Vector, Graph')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tranformation = transformation
        assert ((self.image+self.image_vector+self.vector+self.graph) is 1)

    def get_iterator(self,epoch=0):
        def _load_function(idx):
            idx = idx % len(self.dataset)
            if self.image:
                x = self.dataset[idx]['image']
                return x
            elif self.image_vector:
                pass
                return x, t
            elif self.vector:
                pass
                t = self.dataset[idx]
                return t
            elif self.graph:
                pass
                g = self.dataset[idx]
                return g

        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch) == 1)
            return batch
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self):
        return self.get_iterator()

    def __len__(self):
        raise NotImplementedError()

    def _get_description(self):
        description = {'idx': self.idx,
                       'name': self.name,
                       'size': self.size,
                       'description': self.description,
                       'samples': self.n_samples,
                       'representation': [self.image, self.image_vector, self.vector, self.graph]}
        return description
