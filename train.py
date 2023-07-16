from models import Generator, Discriminator
from data_loader import Customized_CHAOS
from losses import *
from utils import *

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR

import glob
from tqdm import tqdm
import argparse
import copy

def train(args):
    glr = args.glr
    dlr = args.dlr
    betas = (args.betas[0], args.betas[1])
    print(f"glr: {glr} - dlr: {dlr}")
    print(f"betas: {betas}")
    
    curr_epoch = args.curr_epoch
    curr_step = args.curr_step
    curr_batch = args.curr_batch
    device = torch.device(args.device)
    batchsize = args.batchsize
    experiment = args.experiment
    display_step = args.display_step # 4 batch_size (1 step = 2/64 batch_size -> 128 step -> 4 batch_size)
    target_size = 256
    display_imgs = [] # use for display multiple images of multiple batches
        
    set_seed(args.random_seed)
    
    CHAOS_dataset = Customized_CHAOS(path=args.dataset_path, split='train', modals=('t2','ct'))
    CHAOS_dataloader = DataLoader(CHAOS_dataset, batch_size=batchsize, shuffle=True)
    
    genCT2MR = Generator(1, 64, 2, 3, True, True).to(device)
    genCT2MR_use = copy.deepcopy(genCT2MR).to(device)
    genMR2CT = Generator(1, 64, 2, 3, True, True).to(device)
    gen_opt = torch.optim.Adam(list(genCT2MR.parameters()) + list(genMR2CT.parameters()), lr=glr, betas=betas)
    gen_opt.zero_grad()

    scheduler = MultiStepLR(gen_opt,
                            milestones=[18, 25, 35], # List of epoch indices
                            gamma =0.5) # Multiplicative factor of learning rate decay

    # scheduler = StepLR(gen_opt,
    #                    step_size = 3, # Period of learning rate decay
    #                    gamma = 0.8) # Multiplicative factor of learning rate decay

    disc_x_MR = Discriminator(c_dim=1, image_size=256).to(device)
    disc_x_MR_opt = torch.optim.Adam(disc_x_MR.parameters(), lr=dlr, betas=betas)
    disc_x_MR_opt.zero_grad()

    disc_t_MR = Discriminator(c_dim=1, image_size=256).to(device)
    disc_t_MR_opt = torch.optim.Adam(disc_t_MR.parameters(), lr=dlr, betas=betas)
    disc_t_MR_opt.zero_grad()

    disc_t_CT = Discriminator(c_dim=1, image_size=256).to(device)
    disc_t_CT_opt = torch.optim.Adam(disc_t_CT.parameters(), lr=dlr, betas=betas)
    disc_t_CT_opt.zero_grad()

    disc_x_CT = Discriminator(c_dim=1, image_size=256).to(device)
    disc_x_CT_opt = torch.optim.Adam(disc_x_CT.parameters(), lr=dlr, betas=betas)
    disc_x_CT_opt.zero_grad()
    
    genCT2MR.apply(weights_init)
    genCT2MR_use.apply(weights_init)
    genMR2CT.apply(weights_init)
    disc_x_MR.apply(weights_init)
    disc_t_MR.apply(weights_init)
    disc_t_CT.apply(weights_init)
    disc_x_CT.apply(weights_init)

    disc_x_MR_scaler = GradScaler()
    disc_t_MR_scaler = GradScaler()
    disc_x_CT_scaler = GradScaler()
    disc_t_CT_scaler = GradScaler()
    gen_scaler = GradScaler()
    
    ###################################################################################################################################

    mean_discriminator_x_MR_loss = 0
    mean_discriminator_t_MR_loss = 0
    mean_discriminator_x_CT_loss = 0
    mean_discriminator_t_CT_loss = 0

    mean_generator_loss = 0
    mean_generator_cycle_loss = 0
    mean_generator_adversarial_loss = 0
    mean_generator_identity_loss = 0

    #Dictionary for ploting losses 
    dict_loss = {'Learning rate': [], 'Adversarial Loss': [], 'Cycle Loss': [], 'Identity Loss': [],
              'MR-Discriminator of whole image Loss': [], 'MR-Discriminator of tumor image Loss': [],
              'CT-Discriminator of whole image Loss': [], 'CT-Discriminator of tumor image Loss': [],
              'display_step': []}
    dict_loss_step = 1 # x-axis when plotting loss
    
    # create folder for storing synthesis mr images and its prediction
    input_name, output_name = 'input', 'output'
    os.system(f'mkdir {input_name} {output_name}')
    
    # nnUNet_download() # prepare nnUNet-segmentation model for inferencing downstream task on MR-synthesis images
    # Setting environment path for nnUNet
    # os.environ['nnUNet_raw'] = f'{args.nnUNet_dir}/nnUNet_raw'
    # os.environ['nnUNet_preprocessed'] = f'{args.nnUNet_dir}/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = f'{args.nnUNet_dir}/nnUNet_results'
    
    # Getting t2 and ct images for calculating FID score
    ct_img_path = sorted(glob.glob(f'{args.dataset_path}/train/ct/*/DICOM_anon/*.dcm*'))
    ct_batch = torch.cat(list(map(get_ct_img, ct_img_path))).unsqueeze(1)

    t2_img_path = glob.glob(f'{args.dataset_path}/train/t2/*/DICOM_anon/*.dcm')
    t2_real = list(map(get_t2, t2_img_path))
    
    for epoch in range(args.epoch):
        curr_epoch += 1
        
        for (t2_x_real, t2_t_img, (_, t2_mask)),  (ct_x_real, ct_t_img, (_, ct_mask)) in tqdm(CHAOS_dataloader):
            curr_step += 1
            curr_batch += 2

            t2_x_real = t2_x_real.to(device)
            t2_t_img = t2_t_img.to(device)
            t2_mask = t2_mask.to(device)

            ct_x_real = ct_x_real.to(device)
            ct_t_img = ct_t_img.to(device)
            ct_mask = ct_mask.to(device)
            
            ############## Update Discrimnator CT ##############

            with torch.no_grad():
                ct_x_fake, ct_t_fake = genMR2CT(t2_x_real, t2_t_img, mode="train")
            # whole images
            with autocast():
                disc_x_CT_loss = get_disc_loss(ct_x_real, ct_x_fake, disc_x_CT, adv_criterion)
            disc_x_CT_scaler.scale(disc_x_CT_loss).backward()

            if curr_batch % batchsize == 0:
                disc_x_CT_scaler.step(disc_x_CT_opt)
                disc_x_CT_scaler.update()
                disc_x_CT_opt.zero_grad()            
            
            # tumor images
            real_tumor_index = loss_filter(ct_t_img)
            fake_tumor_index = loss_filter(t2_t_img)

            have_turmor_ct_t_img = torch.index_select(ct_t_img, dim=0, index=real_tumor_index)
            have_turmor_ct_t_fake = torch.index_select(ct_t_fake, dim=0, index=fake_tumor_index)

            with autocast():
                disc_t_CT_loss = get_disc_loss(have_turmor_ct_t_img, have_turmor_ct_t_fake, disc_t_CT, adv_criterion)
            disc_t_CT_scaler.scale(disc_t_CT_loss).backward()

            if curr_batch % batchsize == 0:
                disc_t_CT_scaler.step(disc_t_CT_opt)
                disc_t_CT_scaler.update()
                disc_t_CT_opt.zero_grad()    
            
            
            ############## Update Discrimnator MR ##############

            with torch.no_grad():
                t2_x_fake, t2_t_fake = genCT2MR(ct_x_real, ct_t_img, mode="train")
            # whole image
            with autocast():
                disc_x_MR_loss = get_disc_loss(t2_x_real, t2_x_fake, disc_x_MR, adv_criterion)
            disc_x_MR_scaler.scale(disc_x_MR_loss).backward()

            if curr_batch % batchsize == 0:
                disc_x_MR_scaler.step(disc_x_MR_opt)
                disc_x_MR_scaler.update()
                disc_x_MR_opt.zero_grad()

            # tumor image
            real_tumor_index = loss_filter(t2_t_img)
            fake_tumor_index = loss_filter(ct_t_img)

            have_tumor_t2_t_img = torch.index_select(t2_t_img, dim=0, index=real_tumor_index)
            have_tumor_t2_t_fake = torch.index_select(t2_t_fake, dim=0, index=fake_tumor_index)

            with autocast():
                disc_t_MR_loss = get_disc_loss(have_tumor_t2_t_img, have_tumor_t2_t_fake, disc_t_MR, adv_criterion)
            disc_t_MR_scaler.scale(disc_t_MR_loss).backward()

            if curr_batch % batchsize == 0:
                disc_t_MR_scaler.step(disc_t_MR_opt)
                disc_t_MR_scaler.update()
                disc_t_MR_opt.zero_grad()

            ############## Update Generator ##############
            set_requires_grad([disc_x_CT, disc_t_CT, disc_x_MR, disc_t_MR], False)
            
            with autocast():
                gen_loss, gen_adversarial_loss, gen_identity_loss, gen_cycle_loss, fake_ct, fake_t2 = get_gen_loss(ct_x_real, ct_t_img, t2_x_real, t2_t_img,
                                                      genCT2MR, genMR2CT, disc_x_CT, disc_t_CT, disc_x_MR, disc_t_MR,
                                                      adv_criterion=adv_criterion, identity_criterion=recon_criterion,
                                                      cycle_criterion=recon_criterion)
            gen_scaler.scale(gen_loss).backward()      
            if curr_batch % batchsize == 0:
                gen_scaler.step(gen_opt)
                gen_scaler.update()
                gen_opt.zero_grad()
                
            moving_average(genCT2MR, genCT2MR_use, beta=0.999)
            
            set_requires_grad([disc_x_CT, disc_t_CT, disc_x_MR, disc_t_MR], True)
            
            if curr_batch % 1000 == 0:
                curr_batch = 0
            
            # Keep track of the average discriminator loss
            mean_discriminator_x_MR_loss += disc_x_MR_loss.item() / display_step
            mean_discriminator_t_MR_loss += disc_t_MR_loss.item() / display_step
            mean_discriminator_x_CT_loss += disc_x_CT_loss.item() / display_step
            mean_discriminator_t_CT_loss += disc_t_CT_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            mean_generator_adversarial_loss += gen_adversarial_loss.item() / display_step
            mean_generator_cycle_loss += gen_cycle_loss.item() / display_step
            mean_generator_identity_loss += gen_identity_loss.item() / display_step

            if experiment != None:
                experiment.log_metrics({"disc_x_MR": mean_discriminator_x_MR_loss,
                                        "disc_t_MR": mean_discriminator_t_MR_loss,
                                        "disc_x_CT": mean_discriminator_x_CT_loss,
                                        "disc_t_CT": mean_discriminator_t_CT_loss,
                                        "gen": mean_generator_loss}, step=curr_step)

            # Store images to for displaying
            if (curr_step % display_step) > display_step - 4: # Display 3 images
                with autocast():
                    t2_fake = genCT2MR_use(ct_x_real.detach(), mode='test').detach().cpu().to(torch.float32)
                predictions = nnUNet_predict(t2_fake, input_name, output_name)

                display_imgs.append([[torch.cat([t2_x_real.cpu().detach(), ct_x_real.cpu().detach()]),
                                      ct_mask.cpu().detach()],
                                     [torch.cat([fake_ct[0].to(torch.float32).cpu().detach(), t2_fake]),
                                      predictions.cpu().detach()]])

            ############## Visualization Code ##############
            if curr_step % display_step == 0:
                print(f"Epoch: {curr_epoch}, Step: {curr_step}, Generator Loss: {mean_generator_loss} \n\
                Learning rate: {gen_opt.param_groups[0]['lr']} \n\
                Adversarial Loss: {mean_generator_adversarial_loss} \n\
                Cycle Loss: {mean_generator_cycle_loss} \n\
                Identity Loss: {mean_generator_identity_loss} \n\
                MR-Discriminator of whole image Loss: {mean_discriminator_x_MR_loss}\n\
                MR-Discriminator of tumor image Loss: {mean_discriminator_t_MR_loss}\n\
                CT-Discriminator of whole image Loss: {mean_discriminator_x_CT_loss}\n\
                CT-Discriminator of tumor image Loss: {mean_discriminator_t_CT_loss}\n\
                Numb of unique element in t2: {len(np.unique(fake_t2[0].cpu().to(torch.float32).detach()[0].numpy()))}")
                
                #Calculating FID score    
                calculate_FID(genCT2MR_use, ct_batch, t2_real, args.dim)
                
                dict_loss['Learning rate'].append(gen_opt.param_groups[0]['lr'])
                dict_loss['Adversarial Loss'].append(mean_generator_adversarial_loss)
                dict_loss['Cycle Loss'].append(mean_generator_cycle_loss)
                dict_loss['Identity Loss'].append(mean_generator_identity_loss)
                dict_loss['MR-Discriminator of whole image Loss'].append(mean_discriminator_x_MR_loss)
                dict_loss['MR-Discriminator of tumor image Loss'].append(mean_discriminator_t_MR_loss)
                dict_loss['CT-Discriminator of whole image Loss'].append(mean_discriminator_x_CT_loss)
                dict_loss['CT-Discriminator of tumor image Loss'].append(mean_discriminator_t_CT_loss)
                dict_loss['display_step'].append(display_step*dict_loss_step)
                dict_loss_step += 1
                
                
                for real, fake in display_imgs:
                    show_tensor_images(real, size=(1, target_size, target_size))
                    show_tensor_images(fake, size=(1, target_size, target_size))
                display_imgs = []

                mean_discriminator_x_MR_loss = 0
                mean_discriminator_t_MR_loss = 0
                mean_discriminator_x_CT_loss = 0
                mean_discriminator_t_CT_loss = 0

                mean_generator_loss = 0
                mean_generator_adversarial_loss = 0
                mean_generator_cycle_loss = 0
                mean_generator_identity_loss = 0
                
                ##################### Saving Model #####################
            
                MODELS_DRIVE_PATH = f'{args.model_dir}/saved_models_{curr_step}.pt' # Path for saving pytorch's model
                
                if args.auto_delete_model:
                    if os.path.exists(f'{args.model_dir}/saved_models_{curr_step - display_step}.pt'):
                        for a_file in my_drive.ListFile({'q': f"title = 'saved_models_{curr_step - display_step}.pt'"}).GetList():
                            a_file.Trash()
                            print(f'the file "{a_file["title"]}", is about to get deleted permanently.')
                            a_file.Delete()
                            
                torch.save({
                    'current_epoch': curr_epoch,
                    'current_step': curr_step,
                    'current_batch': curr_batch,

                    'disc_x_MR_state_dict': disc_x_MR.state_dict(),
                    'disc_x_MR_opt_state_dict': disc_x_MR_opt.state_dict(),
                    'disc_x_MR_scaler_state_dict': disc_x_MR_scaler.state_dict(),

                    'disc_t_MR_state_dict': disc_t_MR.state_dict(),
                    'disc_t_MR_opt_state_dict': disc_t_MR_opt.state_dict(),
                    'disc_t_MR_scaler_state_dict': disc_t_MR_scaler.state_dict(),

                    'disc_t_CT_state_dict': disc_t_CT.state_dict(),
                    'disc_t_CT_opt_state_dict': disc_t_CT_opt.state_dict(),
                    'disc_t_CT_scaler_state_dict': disc_t_CT_scaler.state_dict(),

                    'disc_x_CT_state_dict': disc_x_CT.state_dict(),
                    'disc_x_CT_opt_state_dict': disc_x_CT_opt.state_dict(),
                    'disc_x_CT_scaler_state_dict': disc_x_CT_scaler.state_dict(),

                    'genCT2MR_state_dict': genCT2MR.state_dict(),
                    'genCT2MR_use_state_dict': genCT2MR_use.state_dict(),
                    'genMR2CT_state_dict': genMR2CT.state_dict(),
                    'gen_opt_state_dict': gen_opt.state_dict(),
                    'gen_scaler_state_dict': gen_scaler.state_dict(),

                    'disc_x_MR_scaler_state_dict': disc_x_MR_scaler.state_dict(),
                    'disc_t_MR_scaler_state_dict': disc_t_MR_scaler.state_dict(),
                    'disc_x_CT_scaler_state_dict': disc_x_CT_scaler.state_dict(),
                    'disc_t_CT_scaler_state_dict': disc_t_CT_scaler.state_dict(),
                    'gen_scaler_state_dict': gen_scaler.state_dict(),

                    'scheduler_state_dict': scheduler.state_dict()

                }, MODELS_DRIVE_PATH)

        scheduler.step()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-glr', type=float, default=1e-4, help='Learning rate of Generator')
    parser.add_argument('-dlr', type=float, default=8e-4, help='Learning rate of Discriminator')
    parser.add_argument('-betas', type=tuple, default=(0.5, 0.9), help='Betas hyperparater')
    
    parser.add_argument('-epoch', type=int, default=100, help='Number of epoch')
    parser.add_argument('-curr_epoch', type=int, default=0, help='Current epoch')
    parser.add_argument('-curr_step', type=int, default=0, help='Current step')
    parser.add_argument('-curr_batch', type=int, default=0, help='Current batch')
    parser.add_argument('-batchsize', type=int, default=2, help='Batchsize')
    parser.add_argument('-experiment', type=str, default=None, help='experiment-API for monitoring the training process')
    parser.add_argument('-display_step', type=int, default=1024, help='Number of steps to display progress')
    
    parser.add_argument('-dim', type=int, default=768, choices=[64, 192, 768, 2048], help="Number of feature's dimesion extracted for calculating FID score")
    
    parser.add_argument('-random_seed', type=int, default='42')
    parser.add_argument('-device', type=str, default='cuda', help='Device using to train model')
    
    parser.add_argument('-dataset_path', type=str, default='CHAOS_preprocessed_v2', help="CHAOS dataset's training path")
    
    parser.add_argument('-model_dir', type=str, default='drive/MyDrive/tarGAN_saved_models', help="Path for saving model")
    parser.add_argument('-auto_delete_model', action='store_true', help="Deleting previous model if you stored it on Drive")
    
    parser.add_argument('-nnUNet_dir', type=str, default='./MRI synthesis from CT', help='folder which stores nnUNet stuff')
    # parser.add_argument('-nnUNet_raw', type=str, default='another_drive/MyDrive/nnUNet/nnUNet_raw', help='environment path for nnUNet')
    # parser.add_argument('-nnUNet_preprocessed', type=str, default='another_drive/MyDrive/nnUNet/nnUNet_preprocessed', help='environment path for nnUNet')
    # parser.add_argument('-nnUNet_results', type=str, default='another_drive/MyDrive/nnUNet/nnUNet_results', help='environment path for nnUNet')
    
    args = parser.parse_args()
    
    train(args)                    
                        
                        
                        
    
