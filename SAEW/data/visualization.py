
import argparse
import glob
import scipy.io as sio

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='./beedance', help='dataset name ')
args = parser.parse_args()
for data_path in glob.glob('%s/*.mat' % (args.data_path)):
    dataset = sio.loadmat(data_path)
    print()
    