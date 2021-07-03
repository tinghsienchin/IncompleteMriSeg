import argparse

parser = argparse.ArgumentParser(description = "DoDNet")

parser.add_argument('--seed', type = int, default = 1234, help = 'random seed')
parser.add_argument('--power', type = float, default = 0.9)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--weight_std', type = bool, default = False)
parser.add_argument('--num_workers', type = int, default = 16)
parser.add_argument('--num_classes', type = int, default = 1)
parser.add_argument('--learning_rate', type = float, default = 1e-3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--reload_from_checkpoint", type=bool, default=False)
parser.add_argument('--mri_size', type=list, default=[128, 128, 128],help='patch size of train samples')
parser.add_argument('--save',default='model1',help='save path of trained model')
parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train (default: 10)')


parser.add_argument("--dataset_path", type=str, default='../fixed_data/')
parser.add_argument("--matpath", type=str, default='../matoutput/')
parser.add_argument("--reload", type=bool , default= False)
parser.add_argument("--reload_path", type=str, default='./output/model1/xx.pth')
parser.add_argument("--early_stop", type=int, default=50)


args = parser.parse_args()