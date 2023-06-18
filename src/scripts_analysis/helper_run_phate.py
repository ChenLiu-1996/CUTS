import argparse
import sys
import warnings

import numpy as np
import phate
from sklearn.preprocessing import normalize

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--phate_path', type=str, required=True)
    parser.add_argument('--random_seed', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    numpy_array = np.load(args.load_path)
    latent = numpy_array['latent']

    phate_op = phate.PHATE(random_state=args.random_seed,
                           n_jobs=args.num_workers,
                           verbose=False)
    data_phate = phate_op.fit_transform(normalize(latent, axis=1))
    with open(args.phate_path, 'wb+') as f:
        np.savez(f, data_phate=data_phate)

    sys.stdout.write('SUCCESS!')
