import torch.nn as nn
import torch

adv_criterion = nn.MSELoss()
# adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()

def loss_filter(mask,device="cuda"):
    list = []
    for i, m in enumerate(mask):
        if torch.any(m != 0):
            list.append(i)
    index = torch.tensor(list, dtype=torch.long).to(device)
    return index

######### DISCRIMINATOR'S LOSS ##########
def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    disc_fake_X_hat = disc_X(fake_X.detach()) # Detach generator
    disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2

    return disc_loss

######### GENERATOR'S LOSS ##########
def get_gen_adversarial_loss(real_x_X, real_t_X, disc_x_Y, disc_t_Y, gen_XY, adv_criterion):
    fake_x_Y, fake_t_Y = gen_XY(real_x_X, real_t_X)
    disc_fake_x_Y_hat = disc_x_Y(fake_x_Y)
    disc_fake_t_Y_hat = disc_t_Y(fake_t_Y)

    adversarial_loss_x = adv_criterion(disc_fake_x_Y_hat, torch.ones_like(disc_fake_x_Y_hat))
    adversarial_loss_t = adv_criterion(disc_fake_t_Y_hat, torch.ones_like(disc_fake_t_Y_hat))

    return adversarial_loss_x + adversarial_loss_t, fake_x_Y, fake_t_Y

def get_identity_loss(real_x_X, real_t_X, gen_YX, identity_criterion):
    identity_x_X, identity_t_X = gen_YX(real_x_X, real_t_X)

    identity_x_loss = identity_criterion(identity_x_X, real_x_X)
    identity_t_loss = identity_criterion(identity_t_X, real_t_X)

    return identity_x_loss + identity_t_loss

def get_cycle_consistency_loss(real_x_X, real_t_X, fake_x_Y, fake_t_Y, gen_YX, cycle_criterion):
    cycle_x_X, cycle_t_X = gen_YX(fake_x_Y, fake_t_Y)

    cycle_loss_x = cycle_criterion(cycle_x_X, real_x_X)
    cycle_loss_t = cycle_criterion(cycle_t_X, real_t_X)

    return cycle_loss_x + cycle_loss_t

def get_cross_loss(fake_x_A, fake_t_A, mask_A, fake_x_B, fake_t_B, mask_B, cross_criterion):
    loss_A = cross_criterion(fake_x_A * mask_A, fake_t_A)
    loss_B = cross_criterion(fake_x_B * mask_B, fake_t_B)

    return loss_A + loss_B

def get_gen_loss(real_x_A, real_t_A, real_x_B, real_t_B, gen_AB, gen_BA, disc_x_A, disc_t_A, disc_x_B, disc_t_B,
                 adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):

    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_BA, fake_x_A, fake_t_A = get_gen_adversarial_loss(real_x_B, real_t_B, disc_x_A, disc_t_A, gen_BA, adv_criterion)
    adv_loss_AB, fake_x_B, fake_t_B = get_gen_adversarial_loss(real_x_A, real_t_A, disc_x_B, disc_t_B, gen_AB, adv_criterion)
    gen_adversarial_loss = adv_loss_BA + adv_loss_AB

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_A = get_identity_loss(real_x_A, real_t_A ,gen_BA, identity_criterion)
    identity_loss_B = get_identity_loss(real_x_B, real_t_B, gen_AB, identity_criterion)
    gen_identity_loss = identity_loss_A + identity_loss_B

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_BA = get_cycle_consistency_loss(real_x_A, real_t_A, fake_x_B, fake_t_B, gen_BA, cycle_criterion)
    cycle_loss_AB = get_cycle_consistency_loss(real_x_B, real_t_B, fake_x_A, fake_t_A, gen_AB, cycle_criterion)
    gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

    # Total loss
    gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss

    return gen_loss, gen_adversarial_loss, gen_identity_loss, gen_cycle_loss, (fake_x_A, fake_t_A) , (fake_x_B, fake_t_B)
