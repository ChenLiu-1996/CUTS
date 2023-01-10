from data_utils.extend import ExtendedDataset
from data_utils.split import split_dataset
from datasets import BerkeleyNaturalImages, BrainVentricles, DiabeticMacularEdema, PolyP, Retina
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, mode: str = 'train'):
    # Read dataset.
    if config.dataset_name == 'retina':
        dataset = Retina(base_path=config.dataset_path)
    elif config.dataset_name == 'berkeley':
        dataset = BerkeleyNaturalImages(base_path=config.dataset_path)
    elif config.dataset_name == 'polyp':
        dataset = PolyP(base_path=config.dataset_path)
    elif config.dataset_name == 'macular_edema':
        dataset = DiabeticMacularEdema(base_path=config.dataset_path)
    elif config.dataset_name == 'brain_ventricles':
        dataset = BrainVentricles(base_path=config.dataset_path)
    else:
        raise Exception(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()
    '''
    Dataset Split.
    Something special here.
    Since the method is unsupervised/self-supervised, we believe it is justifiable to:
        (1) Split the dataset to a train set and a validation set when training the model.
        (2) Use the entire dataset as the test set for evaluating the segmentation performance.
    We believe this is reasonable because ground truth label is NOT used during the training stage.
    '''
    # Load into DataLoader
    if mode == 'train':
        ratios = [float(c) for c in config.train_val_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        train_set, val_set = split_dataset(dataset=dataset,
                                           splits=ratios,
                                           random_seed=config.random_seed)

        min_batch_per_epoch = 5
        desired_len = max(len(train_set),
                          config.batch_size * min_batch_per_epoch)
        train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

        train_set = DataLoader(dataset=train_set,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=config.num_workers)
        val_set = DataLoader(dataset=val_set,
                             batch_size=len(val_set),
                             shuffle=False,
                             num_workers=config.num_workers)
        return train_set, val_set, num_image_channel
    else:
        test_set = DataLoader(dataset=dataset,
                              batch_size=len(dataset),
                              shuffle=False,
                              num_workers=config.num_workers)
        return test_set, num_image_channel


def prepare_dataset_supervised(config: AttributeHashmap):
    # Read dataset.
    if config.dataset_name == 'retina':
        dataset = Retina(base_path=config.dataset_path)
    elif config.dataset_name == 'berkeley':
        dataset = BerkeleyNaturalImages(base_path=config.dataset_path)
    elif config.dataset_name == 'polyp':
        dataset = PolyP(base_path=config.dataset_path)
    elif config.dataset_name == 'macular_edema':
        dataset = DiabeticMacularEdema(base_path=config.dataset_path)
    elif config.dataset_name == 'brain_ventricles':
        dataset = BrainVentricles(base_path=config.dataset_path)
    else:
        raise Exception(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()
    num_classes = dataset.num_classes()

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    train_set, val_set, test_set = split_dataset(
        dataset=dataset, splits=ratios, random_seed=config.random_seed)

    min_batch_per_epoch = 5
    desired_len = max(len(train_set), config.batch_size * min_batch_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=len(val_set),
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=len(test_set),
                          shuffle=False,
                          num_workers=config.num_workers)
    return train_set, val_set, test_set, num_image_channel, num_classes
