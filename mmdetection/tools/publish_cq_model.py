import argparse
import subprocess
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('mode', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, mode):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    in_file_name = os.path.basename(in_file)
    out_file = in_file.replace(in_file_name, mode) + '.pth'
    print(out_file)
    torch.save(checkpoint, out_file)
    # sha = subprocess.check_output(['sha256sum', out_file]).decode()
    # final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    # subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.mode)


if __name__ == '__main__':
    main()
