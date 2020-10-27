#! /usr/bin/env python
from __future__ import division
import os
import argparse

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from skimage.color import lab2rgb

from model import TPN, PCN
from data_loader import *
from util import *



class MySolver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        print(device)
        # Build the model.
        self.build_model(args.mode)

    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        print("Loading %s palette names..." % len(text_data))
        print("Making text dictionary...")

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        return input_dict


    def build_model(self, mode):
        # Data loader.
        self.input_dict = self.prepare_dict()

        # Load pre-trained GloVe embeddings.
        emb_file = os.path.join('./data', 'Color-Hex-vf.pth')
        if os.path.isfile(emb_file):
            W_emb = torch.load(emb_file)
            W_emb = W_emb.to(self.device)
        else:
            print("not found Color-Hex-vf.pth")
            exit()
        

        # Data loader.
        self.test_loader, imsize = test_loader(self.args.dataset, self.args.batch_size, self.input_dict)

        # Load the trained generators.
        self.encoder = TPN.EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                    self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
        self.G_TPN = TPN.AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                    self.args.n_layers, self.args.dropout_p).to(self.device)
        self.G_PCN = PCN.UNet(imsize, self.args.add_L).to(self.device)


    def load_model(self, mode, resume_epoch):
        print('Loading the trained model from epoch {}...'.format(resume_epoch))
        encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
        G_TPN_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))


    def test_TPN(self):
        # Load model.
        
        if self.args.resume_epoch:
            self.load_model(self.args.mode, self.args.resume_epoch)

        self.encoder.eval()
        self.G_TPN.eval()

        print('Start testing...')
        print("Please input text")
        while(True):

            input_text = input()
            if(input_text == ""):
                break

            input_text = input_text.split()

            words_index = []
            for word in input_text:
                if(word in self.input_dict.word2index):
                    words_index.append(self.input_dict.word2index[word])
        
            if(len(words_index) == 0):
                continue

            input_size = len(words_index)
            for i in range(11-input_size):
                words_index.append(0)
            

            batch_size = 5
            txt_embeddings = torch.LongTensor([words_index for _ in range(batch_size)])
            each_input_size = [input_size for _ in range(batch_size)]

            print(txt_embeddings.data)
            print(each_input_size)


            # Prepare test data.
            txt_embeddings = txt_embeddings.to(self.device)

            # Prepare input and output variables.
            palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
            fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

            # ============================== Text-to-Palette ==============================#
            # Condition for the generator.
            encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
            encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

            # Generate color palette.
            for i in range(5):
                palette, decoder_context, decoder_hidden, _ = self.G_TPN(palette,
                                                                        decoder_hidden.squeeze(0),
                                                                        encoder_outputs,
                                                                        each_input_size,
                                                                        i)
                fake_palettes[:, 3 * i:3 * (i + 1)] = palette

            # ================================ Save Results ================================#
            # Input text.
            input_text = ''
            x = 0
            for idx in txt_embeddings[x]:
                if idx.item() == 0: break
                input_text += self.input_dict.index2word[idx.item()] + ' '

            # Save palette generation results.
            fig1, axs1 = plt.subplots(nrows=4, ncols=5)
            axs1[0][0].set_title(input_text + '[fake]')
            for x in range(4):
                for k in range(5):
                    lab = np.array([fake_palettes.data[x][3 * k],
                                fake_palettes.data[x][3 * k + 1],
                                fake_palettes.data[x][3 * k + 2]], dtype='float64')
                    rgb = lab2rgb_1d(lab)
                    axs1[x][k].imshow([[rgb]])
                    axs1[x][k].axis('off')


            plt.show()





def main(args):

    # Solver for training and testing Text2Colors.
    solver = MySolver(args)

    # Train or test.
    solver.test_TPN()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    # text2pal
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--n_layers', type=int, default=1)
    # pal2color
    parser.add_argument('--always_give_global_hint', type=int, default=1)
    parser.add_argument('--add_L', type=int, default=1)

    # Training and testing configuration.
    parser.add_argument('--mode', type=str, default='train_TPN',
                        choices=['train_TPN', 'train_PCN', 'test_TPN', 'test_text2colors'])
    parser.add_argument('--dataset', type=str, default='bird256', choices=['imagenet', 'bird256'])
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('--resume_epoch', type=int, default=1000, help='resume training from this epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--lambda_sL1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_KL', type=float, default=0.5, help='weight for KL loss')
    parser.add_argument('--lambda_GAN', type=float, default=0.1)

    # Directories.
    parser.add_argument('--text2pal_dir', type=str, default='./models/TPN')
    parser.add_argument('--pal2color_dir', type=str, default='./models/PCN')
    parser.add_argument('--train_sample_dir', type=str, default='./samples/train')
    parser.add_argument('--test_sample_dir', type=str, default='./samples/test')

    # Step size.
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many steps to wait before logging training status')
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='how many steps to wait before saving the training output')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='how many steps to wait before saving the trained models')
    args = parser.parse_args()
    print(args)
    main(args)