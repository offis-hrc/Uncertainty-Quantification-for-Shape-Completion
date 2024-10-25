import argparse
import os
import datetime
import random
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from dataset import ShapeNet
from models import PCN_CONFIDENCE as PCN
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, cd_loss_L1_confidence, cd_loss_L1_confidence_diff, emd_loss, emd_loss_original, emd_confidence_loss, emd_confidence_loss_pick, cd_loss_L1_confidence_pred
from visualization import plot_pcd_one_view, plot_pcd_one_view_confidence

torch.cuda.set_device(0)
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')

    train_dataset = ShapeNet('data/PCN', 'train', params.category)
    val_dataset = ShapeNet('data/PCN', 'valid', params.category)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    log(log_fd, "Dataset loaded!")

    # model
    dense_count = 9216
    model = PCN(num_dense=dense_count, latent_dim=1024, grid_size=3).to(params.device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        torch.cuda.empty_cache()
        # hyperparameter alpha
        if train_step < 8000:
            alpha = 0.01
            beta = 0.00
            gamma = 0.03  
        elif train_step < 20000:
            alpha = 0.1 
        elif train_step < 40000:
            alpha = 0.5
        else:
            alpha = 1.0
        #if train_step < 40000:
        #    beta = 0.
        #else:
        #    beta = 10.0
        #    alpha = 0.0
            
        '''
        elif train_step < 50000:
            alpha = 0.1
            beta = 0.0
            gamma = 0.02
        elif train_step < 100000:
            alpha = 0.1
            beta = 0.01
            gamma = 0.02
        elif train_step < 150000:
            alpha = 0.5
            beta = 0.1
            gamma = 0.03
        elif train_step < 200000:
            alpha = 1.0
            beta = 0.5
            gamma = 0.01
        elif train_step < 275000:
            alpha = 1.0
            beta = 0.1
            gamma = 0.01
        elif train_step < 300000:
            alpha = 1.0
            beta = 0.5
            gamma = 0.001
        else:
            alpha = 1.25
            beta = 1.0
            gamma = 0.001
        '''
        #gamma = 0.02 for best performance CD
        gamma = 0.02
        
        

        # training
        model.train()
        for i, (p, c) in enumerate(train_dataloader):
            p, c = p.to(params.device), c.to(params.device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred, confidence = model(p)
            
            # loss function
            if params.coarse_loss == 'cd':
                #loss1 = cd_loss_L1(coarse_pred[:,:,:3], c)
                loss1 = cd_loss_L1_confidence(coarse_pred[:,:,:], c, gamma)
            elif params.coarse_loss == 'emd':
                coarse_c = c[:, :1024, :]
                fine_c = c[:, :, :]
                #loss1 , match = emd_loss_original(coarse_pred[:,:,:3], coarse_c)
                loss1 , match = emd_confidence_loss(coarse_pred, coarse_c, 0.02)#, 0.02) 
                #print('coarse emd',match.size(), loss1)
                #match_array = match.cpu().detach().numpy().reshape((32,-1))
                #match_array = match.cpu().detach().numpy().reshape((32,1024*3))                
                #print(np.shape(match_array))
                
                #match_array = match_array.reshape(())
                #log(log_fd, match)
                #np.savetxt('match.txt', np.around(match_array[2,:], 3))#, delimiter=',')
                #confidence_array = confidence.cpu().detach().numpy().reshape((32,-1)) 
                #np.savetxt('confidence.txt', np.around(confidence_array[2,:], 3))
                #print('match file written!')
            else:
                raise ValueError('Not implemented loss {}'.format(params.coarse_loss))
            
            #dense_pred_pts = dense_pred[:, :, :3]
            #dense_pred_conf =  dense_pred[:, :, 3]
            

            #loss2 = cd_loss_L1(dense_pred, c)
            #gamma = 0.001
            fine_c = c[:, :9216, :]
            #print('c size', c.size(), c[20])
            
            #print('fine size', fine_c_plot.size(), fine_c_plot.type(), fine_c_plot[20])
            #if epoch<50:
            #loss2 = loss1#cd_loss_L1_confidence(dense_pred, fine_c, gamma) #cd_loss_L1(dense_pred[:,:,:3], fine_c)
            #cd_loss_L1_confidence(dense_pred, fine_c, gamma) #cd_loss_L1(dense_pred[:,:,:3], fine_c)
            loss2, match = emd_confidence_loss(dense_pred, fine_c, gamma)
            #loss2 , match = emd_loss_original(dense_pred[:,:,:3], fine_c)
            #loss_conf_pred = cd_loss_L1_confidence_pred(dense_pred, gamma)
            #else:
            #gamma = 0.03
            #loss3 = cd_loss_L1_confidence_diff(dense_pred, fine_c, gamma)
            #loss3 = cd_loss_L1_confidence(dense_pred, fine_c, gamma)
            #loss2, match = emd_loss_original(dense_pred, fine_c)
            #match_array = match.cpu().detach().numpy().reshape((-1,4096,4096))
            #np.savetxt('match.txt', np.around(match_array[2,:,:], 3))
            #if (i+1) % 30 == 0:
            #fine_array = fine_c.cpu().detach().numpy().reshape((32,-1))
            #    dense_array = dense_pred.cpu().detach().numpy().reshape((32,-1))
            #partial_array = p.cpu().detach().numpy().reshape((32,-1))
            #np.savetxt('partial.txt', np.around(partial_array[2,:], 3))
            #np.savetxt('fine_gt.txt', np.around(fine_array[2,:], 3))
            #    np.savetxt('pred.txt', np.around(dense_array[2,:], 3))
            #print('dense match', match.size())
            
            #loss_conf = emd_confidence_loss(dense_pred, match, confidence, 0.05)
            #print('fine emd_conf', loss_conf)
            
            #loss3 = cd_loss_L1(dense_pred, c)

            #loss = loss1 + alpha * loss2 + alpha * loss_conf
            if train_step < 100000:
                conf_weight = train_step / 100000. 
            loss = loss1 + 0.001*alpha*loss2 #+ 3*100*loss_conf_pred + 0.000001*3*100*loss2 #+ 3*100*loss_conf_pred#+ beta*3*100*(loss3) #+ 100*beta*loss3
            #loss = loss1 + 0.1*loss2#loss2
            loss_conf = loss1
            #loss =  loss1 + loss2+ loss_conf


            # back propagation
            loss.backward(retain_graph = False)
            optimizer.step()

            #if (i + 1) % step == 0:
            #    log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense l1 cd = {:.6f}, dense confidence_loss = {:.6f}, total loss = {:.6f}"
            #        .format(epoch, params.epochs, i + 1, len(train_dataloader), loss1.item() * 1e3, loss2.item() * 1e3, loss_conf.item() * 1e3, loss.item() * 1e3))
            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense l1 cd = {:.6f}, dense confidence_loss = {:.6f}, total loss = {:.6f}"
                    .format(epoch, params.epochs, i + 1, len(train_dataloader), loss1.item() , loss2.item() , loss_conf.item() , loss.item() ))
            
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('dense_confidence', loss_conf.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1
        
        lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p, c) in enumerate(val_dataloader):
                p, c = p.to(params.device), c.to(params.device)
                coarse_pred, dense_pred, confidence = model(p)
                fine_c_plot = c[:, :4096, :]
                total_cd_l1 += l1_cd(dense_pred, c).item()

                # save into image
                if rand_iter == i:
                    index = random.randint(0, dense_pred.shape[0] - 1)
                    plot_pcd_one_view_confidence(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [p[index].detach().cpu().numpy(), coarse_pred[index].detach().cpu().numpy(), dense_pred[index].detach().cpu().numpy(), fine_c_plot[index].detach().cpu().numpy()],
                                      ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            total_cd_l1 /= len(val_dataset)
            val_writer.add_scalar('l1_cd', total_cd_l1, val_step)
            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, params.epochs, total_cd_l1 ))
        
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_l1_cd.pth'))
        
        #if total_cd_l1 < best_cd_l1 or :
            #best_epoch_l1 = epoch
            #best_cd_l1 = total_cd_l1
        if  epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'epoch_'+str(epoch)+'.pth'))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pth'))      
    log(log_fd, 'Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))
    log_fd.close()
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    params = parser.parse_args()
    
    train(params)
