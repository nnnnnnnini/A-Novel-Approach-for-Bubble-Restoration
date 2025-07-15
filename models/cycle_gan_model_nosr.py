import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from . import networks
'''
try:
    from apex import amp
except ImportError as error:
    print(error)
'''
import util.util as util

##no sr loss
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True)  # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)


        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            ## 权重原为0.5
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        ## 新增
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,1', help='compute NCE loss on which layers')#'0,4,8,12,16'
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        ## 权重原为0.03 没有sr 权重设为0
        parser.add_argument('--self_regularization', type=float, default=0.0,
                            help='loss between input and generated image')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        ## 修改 新增了ncea,necb两个损失
        self.loss_names = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'NCE_A', 'NCE_B']#
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        #if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #    visual_names_A.append('idt_B')
        #    visual_names_B.append('idt_A')
        ## 以上三行修改为
        if opt.nce_idt and self.isTrain and self.opt.lambda_identity > 0.0:
            self.loss_names += ['NCE_Y', 'NCE_X']# , 'NCE_X'
            visual_names_A += ['idt_A']
            visual_names_B += ['idt_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        ## 修改 加入netF
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','F_A', 'F_B']#
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        ## 新增网络F定义
        self.netF_A = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_B = networks.define_F(opt.output_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            ## 新增nce
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
        #self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        #self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        #self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        #self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        ## 以上四行修改为以下内容
        self.real1 = torch.cat((self.real_A, self.real_B),
                               dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.real2 = torch.cat((self.real_B, self.real_A),
                               dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real1 = torch.flip(self.real1, [3])
                self.real2 = torch.flip(self.real2, [3])
        self.fake1 = self.netG_A(self.real1)
        self.fake2 = self.netG_B(self.real2)
        self.fake_B = self.fake1[:self.real_A.size(0)]
        self.fake_A = self.fake2[:self.real_B.size(0)]
        if self.opt.nce_idt:
            ## self.idt_B = self.fake[self.real_A.size(0):]
            self.idt_B = self.fake1[self.real_A.size(0):]
            self.idt_A = self.fake2[self.real_B.size(0):]

        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

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
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
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

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True).mean()
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True).mean()
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A).mean() * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B).mean() * lambda_B
        # combined loss and calculate gradients
        ## 新增
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
        ## 修改 新加入 nce和 sr loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B +  loss_NCE_A_both  + loss_NCE_B_both + self.loss_SR_A  + self.loss_SR_B + self.loss_idt_A + self.loss_idt_B
        '''
        if self.opt.amp:
            with amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
        '''
        self.loss_G.backward()

    def data_dependent_initialize(self,data):
        ##全部为新增
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G_A(A)，G_B(B)
        if self.opt.isTrain:
            ##先生成器损失还是先判别器
            self.backward_G()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B

            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters() ,self.netF_B.parameters()),
                                                    lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

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

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        ## 新增 F相关的梯度内容
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    ## 新增nce和sr函数
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

    def calculate_SR_loss(self,src,tgt):
        #rgb_mean_src = torch.mean(src,dim=1) #mean over color channel
        #rgb_mean_tgt = torch.mean(tgt,dim=1)
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2

