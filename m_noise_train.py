import argparse
import json
import os
import random
import time
import numpy as np
import torch
import datetime


from model import DeepSpeech
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from visdom import Visdom

parser = argparse.ArgumentParser(description='Noise m training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/libri_train_manifest_only_one.csv')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='models/deepspeech_final.pth', help='Continue from checkpoint model')
parser.add_argument('--iters', default=3, type=int, help='Number of updating m')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')   # 0.4
parser.add_argument('--noise-min', default=0.5,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.8,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)  # 0.5
parser.add_argument('--lamda', default=1, type=float, help='value of lamda')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_current_time():
    return datetime.datetime.now().strftime("%m-%d-%H-%M")


def to_np(x):
    return x.cpu().numpy()


def self_loss(f, f_star, m, lamda, beta=1):
    loss_1 = beta * torch.sum(torch.pow(f - f_star, 2))
    loss_2 = -lamda * torch.sum(torch.log(m))  # / (m.size()[0] * m.size()[1])
    totloss = loss_1 + loss_2
    return totloss, loss_1, loss_2


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Iters',
                    ylabel=var_name)
            )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


class M_Noise_Deepspeech(torch.nn.Module):
    def __init__(self, pkg, input_size):
        super(M_Noise_Deepspeech, self).__init__()
        self.small_const = 1e-6
        self.K = input_size[2]
        self.T = input_size[3]
        self.m = torch.nn.Parameter(torch.Tensor(1 * np.ones((self.K, self.T, 1), dtype=np.float32)).cuda(),
                                    requires_grad=True)
        self.range1 = torch.Tensor(
            np.array(list(range(self.K)) * self.K * self.T).reshape((self.K, self.T, self.K))).cuda()  # range in 2 dim
        self.range2 = torch.Tensor(
            np.array(list(range(self.K)) * self.K * self.T).reshape(
                (self.K, self.T, self.K)).transpose()).cuda()  # range in 0 dim

        self.relu = torch.nn.ReLU()
        self.deepspeech_net = DeepSpeech.load_model_package(pkg)

    def forward(self, input, input_length):
        abs_m = torch.abs(self.m)
        m_tile = abs_m.repeat([1, 1, self.K])
        out = self.relu(m_tile - torch.abs(self.range1 - self.range2)) / (torch.pow(m_tile, 2) + self.small_const)
        blar = (torch.mul(out, (m_tile > 1).float()) + torch.mul((m_tile <= 1).float(),
                                                                 (self.range1 == self.range2).float())).cuda()
        norm_index = torch.sum(blar, dim=2).reshape([self.K, self.T, 1]).repeat([1, 1, self.K])
        blar = blar / norm_index
        inputtile = input[0, 0, :, :].reshape([self.K, self.T, 1]).repeat(1, 1, self.K).cuda()
        inputs = torch.sum(torch.mul(blar, inputtile), dim=0).transpose(0, 1).reshape([1, 1, self.K, self.T])
        y = self.deepspeech_net(inputs, input_length)
        return y


if __name__ == '__main__':
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    loss_results, cer_results, wer_results = torch.Tensor(args.iters), torch.Tensor(args.iters), torch.Tensor(args.iters)

    # best_wer = None
    # use visdom to logging
    visdom_logger = VisdomLinePlotter(env_name='m_noise_trainer')

    avg_loss, start_iter, optim_state = 0, 0, None

    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        print("Loading label from %s" % args.labels_path)
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
    else:
        print("Must load model!")
        exit()

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=False)
    train_sampler = BucketingSampler(train_dataset, batch_size=1)
    train_loader = AudioDataLoader(train_dataset, batch_sampler=train_sampler)

    # get the previous output f* & have modified the forward function
    with torch.no_grad():
        for i, (data) in enumerate(train_loader):  # just once
            # inputs: Nx1xKxT
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            print('input size is:', inputs.size())
            # initial M_Noise model
            M_model = M_Noise_Deepspeech(package, inputs.size())
            M_model.to(device)
            # no update to these parameters
            for para in M_model.deepspeech_net.parameters():
                para.requires_grad = False

            out_star, output_sizes = M_model.deepspeech_net(inputs, input_sizes)
            float_out_star = out_star.transpose(0, 1).float()  # TxNxH
            print('out_star size is:', float_out_star.size())

    parameters = filter(lambda p: p.requires_grad, M_model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=1e-5)

    print(M_model)

    # check whether the deepspeech2 parameters changed
    preParams = {}
    for name, param in M_model.deepspeech_net.named_parameters():
        preParams[name] = to_np(param)

    iter_time = AverageMeter()
    losses = AverageMeter()
    ##
    lamda = args.lamda

    result_dir = './noise_result/' + get_current_time() + '/'
    if not (os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)

    saved_loss = np.ndarray((args.iters), dtype=np.float32)
    saved_m_mean = np.ndarray((args.iters), dtype=np.float32)
    saved_m_std = np.ndarray((args.iters), dtype=np.float32)

    for i in range(start_iter, args.iters):
        M_model.train()
        end = time.time()  # current time
        start_iter_time = time.time()

        for j, (data) in enumerate(train_loader):
            # inputs: Nx1xKxT
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            

            out, output_sizes = M_model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            float_out = out.float()  # ensure float32 for loss

            loss, loss_1, loss_2 = self_loss(float_out, float_out_star, torch.abs(M_model.m), lamda)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_value = loss.item()
            m_mean = torch.mean(M_model.m).item()
            m_std = torch.std(M_model.m).item()

            optimizer.zero_grad()
            # compute gradient
            #loss.backward(retain_graph=True)  # save middle variables
            loss_1.backward(retain_graph=True)
            print('loss_1 grad m is:', M_model.m.grad)
            M_model.m.grad.data.zero_() # avoid grad accumalate

            loss_2.backward(retain_graph=True)
            print('loss_2 grad m is:', M_model.m.grad)
            M_model.m.grad.data.zero_()

            loss.backward()
            print('loss grad m is:', M_model.m.grad)


            # updata parameters
            optimizer.step()
            avg_loss += loss_value

            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            iter_time.update(time.time() - end)
            end = time.time()

            if not args.silent:
                print('Iter: [{0}][{1}]\t'
                      'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} [{loss_1:.4f}] [{loss_2:.4f}]\t'
                      'm avg: {mean:.4f} std: {std:.4f}'.format(
                    (i + 1), (j + 1), iter_time=iter_time, loss=losses, loss_1=loss_1.item(), loss_2=loss_2.item(),
                    mean=m_mean, std=m_std))

                visdom_logger.plot('loss', 'loss_1', 'Class Loss', i, loss_1.item())
                visdom_logger.plot('loss', 'loss_2', 'Class Loss', i, -loss_2.item())
                visdom_logger.plot('loss', 'train', 'Class Loss', i, loss_value)
                visdom_logger.plot('m_mean', 'train', 'M', i, m_mean)
                visdom_logger.plot('m_std', 'train', 'M', i, m_std)

            saved_loss[i] = loss_value
            saved_m_mean[i] = m_mean
            saved_m_std[i] = m_std
            del loss, out, float_out

        losses.reset()

    # the trainable parameter
    for name, param in M_model.named_parameters():
        if param.requires_grad:
            print(name, 'last grad is:')
            print(param.grad)

    # check whether the deepspeech2 parameters changed
    paramDiff = 0.0
    for name, param in M_model.deepspeech_net.named_parameters():
        paramDiff += np.sum(to_np(param) - preParams[name])
    print('Deepspeech2 net parameters change:', paramDiff)
    

    print('Finish training\n'
          'saved m to {}\n'
          'saved fft to {}\n'
          'saved loss to {}\n'.format(
        result_dir + 'm.txt', result_dir + 'fft.txt', result_dir + 'loss.txt'
    ))

    np.savetxt(result_dir + 'fstar.txt', float_out_star.cpu().detach().numpy().flatten())
    np.savetxt(result_dir + 'm.txt', M_model.m.cpu().detach().numpy().reshape(M_model.K, M_model.T))
    np.savetxt(result_dir + 'fft.txt', to_np(inputs).reshape(M_model.K, M_model.T))
    np.savetxt(result_dir + 'loss.txt', saved_loss)
    np.savetxt(result_dir + 'm_mean.txt', saved_m_mean)
    np.savetxt(result_dir + 'm_std.txt', saved_m_std)
