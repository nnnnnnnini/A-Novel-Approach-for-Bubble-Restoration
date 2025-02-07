import numpy as np
import torch
import itertools
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
try:
    from apex import amp
except ImportError as error:
    print(error)



class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,1', help='compute NCE loss on which layers')#'0,4,8,12,16'
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--self_regularization', type=float, default=0.03, help='loss between input and generated image')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        ### 新增
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')


        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        ##self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        ## 修改 'D_real', 'D_fake'即为D_Y  'G_GAN'即为G_A  G为重建损失（G(B)=B）加入GA,GB，NCE变为NCE_A,NCE_B.
        ## 加不加G
        self.loss_names = ['D_Y', 'G_A', 'cycle_A', 'D_X', 'G_B', 'cycle_B', 'NCE_A', 'NCE_B', "G"]
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'rec_A', 'fake_A', 'rec_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        ## 修改 加入self.opt.lambda_identity > 0.0 及
        if opt.nce_idt and self.isTrain and self.opt.lambda_identity > 0.0:
            ## self.loss_names += ['NCE_Y']
            ## self.visual_names += ['idt_B']
            self.loss_names += ['NCE_Y', 'NCE_X']
            self.visual_names += ['idt_B', 'idt_A']

        if self.isTrain:
            ## 修改 self.model_names = ['G', 'F', 'D']
            self.model_names = ['F_A', 'F_B', 'G_A', 'G_B', 'D_Y', 'D_X']
        else:  # during test time, only load G
            ## 修改 self.model_names = ['G']
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        ## 修改
        ## self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        ##self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_A = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_B = networks.define_F(opt.output_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if self.isTrain:
            ##self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_Y = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)
            self.netD_X = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)


            ## 新增
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            ## 新增
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            #self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            #self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            ## 修改
            ## self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            ## self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Y.parameters(), self.netD_X.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G_A(A)，G_B(B)
        if self.opt.isTrain:
            ##先生成器损失还是先判别器
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.compute_G_loss()  # calculate graidents for G

            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netF_B.parameters()),
                                                    lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        ##要不要加.mean()
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        '''
        if self.opt.amp:
            with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
        '''
        loss_D.backward()

        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # 1 fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_Y = self.backward_D_basic(self.netD_Y, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # 1 fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A = self.fake_A
        self.loss_D_X = self.backward_D_basic(self.netD_X, self.real_A, fake_A)

    def optimize_parameters(self):
        # forward
        self.forward()
        ## 先更新g还是d
        # update D
        self.set_requires_grad([self.netD_Y, self.netD_X], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netD_Y, self.netD_X], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.compute_G_loss()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ## 修改self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.real1 = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.real2 = torch.cat((self.real_B, self.real_A), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                ## self.real = torch.flip(self.real, [3])
                self.real1 = torch.flip(self.real1, [3])
                self.real2 = torch.flip(self.real2, [3])

        ## 修改
        ## self.fake = self.netG(self.real)
        ## self.fake_B = self.fake[:self.real_A.size(0)]
        self.fake1 = self.netG_A(self.real1)
        self.fake2 = self.netG_B(self.real2)
        self.fake_B = self.fake1[:self.real_A.size(0)]
        self.fake_A = self.fake2[:self.real_B.size(0)]
        if self.opt.nce_idt:
            ## self.idt_B = self.fake[self.real_A.size(0):]
            self.idt_B = self.fake1[self.real_A.size(0):]
            self.idt_A = self.fake2[self.real_B.size(0):]

        ## 新增
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        '''
        # Fake; stop backprop to the generator by detaching fake_B
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        '''
        ## 修改
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B_pool.query(self.fake_B).detach()
        fake_A = self.fake_A_pool.query(self.fake_A).detach()
        pred_fake_B = self.netD_Y(fake_B)
        pred_fake_A = self.netD_X(fake_A)
        self.loss_D_Y_fake = self.criterionGAN(pred_fake_B, False).mean()
        self.loss_D_X_fake = self.criterionGAN(pred_fake_A, False).mean()
        # Real
        self.pred_real_B = self.netD_Y(self.real_B)
        self.pred_real_A = self.netD_X(self.real_A)
        loss_D_Y_real = self.criterionGAN(self.pred_real_B, True)
        loss_D_X_real = self.criterionGAN(self.pred_real_A, True)
        self.loss_D_Y_real = loss_D_Y_real.mean()
        self.loss_D_X_real = loss_D_X_real.mean()

        # combine loss and calculate gradients
        self.loss_D_Y = (self.loss_D_Y_fake + self.loss_D_Y_real) * 0.5
        self.loss_D_X = (self.loss_D_X_fake + self.loss_D_X_real) * 0.5


        ## 新增
        '''
        if self.opt.amp:
            with amp.scale_loss(self.loss_D_Y, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
            with amp.scale_loss(self.loss_D_X, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
        '''
        self.loss_D_Y.backward()
        self.loss_D_X.backword()

    def calculate_SR_loss(self,src,tgt):
        #rgb_mean_src = torch.mean(src,dim=1) #mean over color channel
        #rgb_mean_tgt = torch.mean(tgt,dim=1)
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""

        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        '''
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        '''
        '''
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.self_regularization > 0.0:
            self.loss_SR = self.opt.self_regularization * self.calculate_SR_loss(self.real_A, self.fake_B)
        else:
            self.loss_SR = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SR
        '''
        ## 修改
        fake_B = self.fake_B
        fake_A = self.fake_A
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake_B = self.netD_Y(fake_B)
            pred_fake_A = self.netD_X(fake_A)
            ##是否加.mean()
            self.loss_G_A = self.criterionGAN(pred_fake_B, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fake_A, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        ## 新增
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A).mean() * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B).mean() * lambda_B

        ## 修改
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE_A = self.calculate_NCE1_loss(self.real_A, self.fake_B)
            self.loss_NCE_B = self.calculate_NCE2_loss(self.real_B, self.fake_A)
        else:
            self.loss_NCE_A, self.loss_NCE_A_bd = 0.0, 0.0
            self.loss_NCE_B, self.loss_NCE_B_bd = 0.0, 0.0

        if self.opt.self_regularization > 0.0:
            self.loss_SR_A = self.opt.self_regularization * self.calculate_SR_loss(self.real_A, self.fake_B)
            self.loss_SR_B = self.opt.self_regularization * self.calculate_SR_loss(self.real_B, self.fake_A)
        else:
            self.loss_SR_A = 0.0
            self.loss_SR_B = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE1_loss(self.real_B, self.idt_B)
            self.loss_NCE_X = self.calculate_NCE2_loss(self.real_A, self.idt_A)
            loss_NCE_A_both = (self.loss_NCE_A + self.loss_NCE_Y) * 0.5
            loss_NCE_B_both = (self.loss_NCE_B + self.loss_NCE_X) * 0.5
        else:
            loss_NCE_A_both = self.loss_NCE_A
            loss_NCE_B_both = self.loss_NCE_B

        self.loss_G = self.loss_G_A + self.loss_G_B+ loss_NCE_A_both + loss_NCE_B_both + self.loss_SR_A  + self.loss_SR_B + self.loss_cycle_A + self.loss_cycle_B
        '''
        if self.opt.amp:
            with amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
        '''
        self.loss_G.backward()

    def calculate_NCE1_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_A(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_A(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_NCE2_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_B(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_B(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals


