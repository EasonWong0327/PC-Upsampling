import open3d as o3d
import time, os, sys, glob, argparse
import numpy as np
import torch
import MinkowskiEngine as ME
import multiprocessing as mp
from model.Network import MyNet

from data import make_data_loader
from utils.loss import get_metrics
from utils.pc_error_wrapper import pc_error

from tensorboardX import SummaryWriter
import logging

import GPUtil
import pynvml
pynvml.nvmlInit()
gpu_index = 0

def print_gpu_memory(device=None):
    if device is None:
        device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    print(f"GPU {device}:")
    print(f"  Total Memory: {mem_info.total / (1024 ** 2):.2f} MB")
    print(f"  Used Memory: {mem_info.used / (1024 ** 2):.2f} MB")
    print(f"  Free Memory: {mem_info.free / (1024 ** 2):.2f} MB")
    print(f"  GPU Utilization: {utilization.gpu}%")
    print(f"  Memory Utilization: {utilization.memory}%")
    print(f"  Temperature: {temperature} C")
    print("-" * 30)

def getlogger(logdir):
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)
  
  handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
  handler.setFormatter(formatter)
  
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  
  logger.addHandler(handler)
  logger.addHandler(console)
  
  return logger


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dataset", default='/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldier/Ply/')
  parser.add_argument("--downsample", default=8, help='Downsample Rate')
  parser.add_argument("--num_test", type=int, default=60, help='how many of the dataset use for testing')
  # parser.add_argument("--dataset_8i", default='/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldierless/Ply/')
  # parser.add_argument("--dataset_8i_GT", default='/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldierless/Ply/')
  # parser.add_argument("--dataset_8i_GT", default='/home/jupyter-eason/data/software/mpeg-pcc-tmc2-master/output0911/soldier_r1/')
  parser.add_argument("--last_kernel_size", type=int, default=5, help='The final layer kernel size, coordinates get expanded by this')

  parser.add_argument("--init_ckpt", default='')
  parser.add_argument("--reset", default=False, action='store_true', help='reset training')

  parser.add_argument("--lr", type=float, default=8e-4)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--global_step", type=int, default=int(1000))
  parser.add_argument("--base_step", type=int, default=int(100),  help='frequency for recording state.')
  parser.add_argument("--test_step", type=int, default=int(200),  help='frequency for test and save.')
  # parser.add_argument("--random_seed", type=int, default=4, help='random_seed.')

  parser.add_argument("--max_norm", type=float, default=1.,  help='max norm for gradient clip, close if 0')
  parser.add_argument("--clip_value", type=float, default=0,  help='max value for gradient clip, close if 0')

  parser.add_argument("--logdir", type=str, default='logs', help="logger direction.")
  parser.add_argument("--ckptdir", type=str, default='ckpts', help="ckpts direction.")
  parser.add_argument("--prefix", type=str, default='8x_0x_ks7', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--lr_gamma", type=float, default=0.5, help="gamma for lr_scheduler.")
  parser.add_argument("--lr_step", type=int, default=200, help="step for adjusting lr_scheduler.")

  args = parser.parse_args()
  return args



def kdtree_partition2(pc, pc2, max_num):
    parts = []
    parts2 = []
    
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
            
    def createKDTree(root, data, data2):
        if len(data) <= max_num:
            parts.append(data)
            parts2.append(data2)
            return
		
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        data2_sorted = data2[np.lexsort(data2.T[dim_index, None])]
		
        point = data_sorted[int(len(data)/2)]  
  		
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))], data2_sorted[: int((len(data2) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):], data2_sorted[int((len(data2) / 2)):]) 
        
        return root
    
    init_root = KD_node(None)
    _ = createKDTree(init_root, pc, pc2)
	
    return parts, parts2



### This testing function is made for larger point clouds and uses kd_tree partition.
def test2(model, test_dataloader, logger, writer, writername, step, test_pc_error, args, device, First_eval):

  start_time = time.time()

  # data.
  test_iter = iter(test_dataloader)

  # loss & metrics.
  sum_loss = 0.
  all_metrics = np.zeros((1,3))
  all_pc_errors = np.zeros(3)
  all_pc_errors2 = np.zeros(2)

  # model & crit.
  model.to(device)# to cpu.
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # loop per batch.
  for i in range(len(test_iter)):
    coords, _, coords_T = test_iter.next()
    parts_pc, parts_pc2 = kdtree_partition2(coords[:,1:].numpy(), coords_T[:,1:].numpy(), 70000)
    out_l = []
    out_cls_l = []
    target_l = []
    keep_l = []
    
    # Forward.
    for j,pc in enumerate(parts_pc):
        p = ME.utils.batched_coordinates([pc])
        p2 = ME.utils.batched_coordinates([parts_pc2[j]])
        f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
        x1 = ME.SparseTensor(features=f.to(device), coordinates=p.to(device))
        
        with torch.no_grad():
            out, out_cls, target, keep = model(x1, coords_T=p2, device=device, prune=True)
        
        out_l.append(out.C[:,1:])
        out_cls_l.append(out_cls.F)
        target_l.append(target)
        keep_l.append(keep)
        
    rec_pc = torch.cat(out_l, 0)
    rec_pc_cls = torch.cat(out_cls_l, 0)
    rec_target = torch.cat(target_l, 0)
    rec_keep = torch.cat(keep_l, 0)
    
    loss = crit(rec_pc_cls.squeeze(), rec_target.type(out_cls.F.dtype).to(device))
    metrics = get_metrics(rec_keep, rec_target)
    
    # get pc_error.
    work_path = '/home/jupyter-eason/project/upsampling/PointCloudUpsampling/'
    if test_pc_error:
      GT_pcd = o3d.geometry.PointCloud()
      GT_pcd.points = o3d.utility.Vector3dVector(coords_T[:,1:])
      # ori_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
      
      GTfile = work_path + 'tmp/'+args.prefix+'GT.ply'
      o3d.io.write_point_cloud(GTfile, GT_pcd, write_ascii=True)
      
      rec_pcd = o3d.geometry.PointCloud()
      logger.info(type(rec_pc))
      logger.info(rec_pc.shape)
      
      rec_pc_np = rec_pc.detach().cpu().numpy()
      rec_pcd.points = o3d.utility.Vector3dVector(rec_pc_np)
      
      recfile = work_path + 'tmp/'+args.prefix+'rec.ply'
      o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
      
      pc_error_metrics = pc_error(infile1=GTfile, infile2=recfile, res=1024)
      pc_error_metrics.to_excel('pc_error_metrics.xlsx')
      pc_errors = [pc_error_metrics['mse1,PSNR (p2point)'][0], 
                  pc_error_metrics['mse2,PSNR (p2point)'][0], 
                  pc_error_metrics['mseF,PSNR (p2point)'][0]]
      
      if First_eval:
          in_pcd = o3d.geometry.PointCloud()
          in_pcd.points = o3d.utility.Vector3dVector(coords[:,1:].numpy())
          
          infile = work_path + 'tmp/'+args.prefix+'in.ply'
          o3d.io.write_point_cloud(infile, in_pcd, write_ascii=True)
          
          pc_error_metrics = pc_error(infile1=GTfile, infile2=infile, res=1024)
          pc_errors2 = [pc_error_metrics['mse1,PSNR (p2point)'][0],
                  pc_error_metrics['mseF,PSNR (p2point)'][0]]
          
    # record.
    with torch.no_grad():
      sum_loss += loss.item()
      all_metrics += np.array(metrics)
      if test_pc_error:
        all_pc_errors += np.array(pc_errors)
        if First_eval:
            all_pc_errors2 += np.array(pc_errors2)

  print('======testing time:', round(time.time() - start_time, 4), 's')

  sum_loss /= len(test_iter)
  all_metrics /= len(test_iter)
  if test_pc_error:
    all_pc_errors /= len(test_iter)
    if First_eval:
        all_pc_errors2 /= len(test_iter)

  # logger.
  logger.info(f'\nIteration: {step}')
  logger.info(f'Sum Loss: {sum_loss:.4f}')
  logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')
  if test_pc_error:
    logger.info(f'all_pc_errors: {np.round(all_pc_errors, 4).tolist()}')
  
  # writer.
  writer.add_scalars(main_tag=writername+'/losses', 
                    tag_scalar_dict={'sum_loss': sum_loss}, 
                    global_step=step)
  
  writer.add_scalars(main_tag=writername+'/metrics', 
                    tag_scalar_dict={'Precision': all_metrics[0,0],
                                     'Recall': all_metrics[0,1],
                                     'IoU': all_metrics[0,2]},
                    global_step=step)
  
  if test_pc_error:
    writer.add_scalars(main_tag=writername+'/out_pc_errors', 
                  tag_scalar_dict={'p2point1': all_pc_errors[0],
                                    'p2point2': all_pc_errors[1],
                                    'p2pointF': all_pc_errors[2],},
                  global_step=step)
    if First_eval:
        writer.add_scalars(main_tag=writername+'/In_pc_errors', 
                      tag_scalar_dict={'p2point1': all_pc_errors2[0],
                                        'p2pointF': all_pc_errors2[1]},
                      global_step=step)
  return


### This testing function is made for smaller point clouds and does not use kd_tree partition.
def test1(model, test_dataloader, logger, writer, writername, step, args, device):

  start_time = time.time()

  # data.
  test_iter = iter(test_dataloader)

  # loss & metrics.
  sum_loss = 0.
  all_metrics = np.zeros((1,3))

  # model & crit.
  model.to(device)# to cpu.
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # loop per batch.
  for i in range(len(test_iter)):
    coords, feats, coords_T = test_iter.next()
    x = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device))
    
    # Forward.
    with torch.no_grad():
        out, out_cls, target, keep = model(x, coords_T=coords_T, device=device, prune=True)
    
    loss = crit(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
    metrics = get_metrics(keep, target)
    
    # record.
    with torch.no_grad():
      sum_loss += loss.item()
      all_metrics += np.array(metrics)

  print('======testing time:', round(time.time() - start_time, 4), 's')

  sum_loss /= len(test_iter)
  all_metrics /= len(test_iter)

  # logger.
  logger.info(f'\nIteration: {step}')
  logger.info(f'Sum Loss: {sum_loss:.4f}')
  logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')

  # writer.
  writer.add_scalars(main_tag=writername+'/losses', 
                    tag_scalar_dict={'sum_loss': sum_loss}, 
                    global_step=step)

  writer.add_scalars(main_tag=writername+'/metrics', 
                    tag_scalar_dict={'Precision': all_metrics[0,0],
                                     'Recall': all_metrics[0,1],
                                     'IoU': all_metrics[0,2]},
                    global_step=step)    
  return


def train(model, train_dataloader, test_dataloader, logger, writer, args, device):
  # Optimizer.
  optimizer = torch.optim.Adam([{"params":model.parameters(), 'lr':args.lr}], 
                                betas=(0.9, 0.999), weight_decay=1e-4)
  # adjust lr.
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # define checkpoints direction.
  ckptdir = os.path.join(args.ckptdir, args.prefix)
  if not os.path.exists(ckptdir):
    logger.info(f'Make direction for saving checkpoints: {ckptdir}')
    os.makedirs(ckptdir)

  # Load checkpoints.
  start_step = 1
  if args.init_ckpt == '':
    logger.info('Random initialization.')
    First_eval = True
  else:
    # load params from checkpoints.
    logger.info(f'Load checkpoint from {args.init_ckpt}')
    ckpt = torch.load(args.init_ckpt)
    model.load_state_dict(ckpt['model'])
    First_eval = False
    # load start step & optimizer.
    if not args.reset:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            print("Optimizer State Load Failed")    
        start_step = ckpt['step'] + 1

  # start step.
  logger.info(f'LR: {scheduler.get_lr()}')
  print('==============', start_step)

  train_iter = iter(train_dataloader)
  start_time = time.time()
  sum_loss = 0
  all_metrics = np.zeros((1,3))
  

  for i in range(start_step, args.global_step+1):
    if i%10==0:
        print(i)
    optimizer.zero_grad()
    
    s = time.time()
    coords, feats, pc_data = train_iter.next()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(coords.shape,feats.shape,pc_data.shape)
    dataloader_time = time.time() - s

    print("After data loading:")
    print_gpu_memory(gpu_index)

    x = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device))
    y = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device))

    print('after SparseTensor,X.shape:',x.shape)
    print(f"After SparseTensor creation:")
    print_gpu_memory(gpu_index)

    if x.__len__() >= 1e10:
      logger.info(f'\n\n\n======= larger than 1e10 ======: {x.__len__()}\n\n\n')
      continue
  
    # Forward.
    ''''
    up-sampling：
    out_cls, target, keep = torch.Size([18799, 1]) torch.Size([18799]) torch.Size([18799])
    out_cls:
                SparseTensor(
              coordinates=tensor([[ 0, 75, 84, 94],
                    [ 0, 35, 87, 24],
                    [ 0, 25, 20, 32],
                    ...,
                    [ 3, 42, 37, 78],
                    [ 3, 80, 79, 40],
                    [ 3, 75, 65, 57]], device='cuda:0', dtype=torch.int32)
              features=tensor([[1.2300],
                    [3.3449],
                    [1.3790],
                    ...,
                    [2.0339],
                    [1.1265],
                    [1.3687]], device='cuda:0', grad_fn=<AddBackward0>)
              coordinate_map_key=coordinate map key:[1, 1, 1]
              coordinate_manager=CoordinateMapManagerGPU_c10(
                    [0, 0, 0]:      CoordinateMapGPU:4x4
                    [1, 1, 1]:      CoordinateMapGPU:18799x4
                    [2, 2, 2]:      CoordinateMapGPU:14568x4
                    [4, 4, 4]:      CoordinateMapGPU:5974x4
                    [8, 8, 8]:      CoordinateMapGPU:1580x4
                    [16, 16, 16]:   CoordinateMapGPU:415x4
                    [2, 2, 2]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:125, kernel map size:106338
                    [4, 4, 4]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:39616
                    [8, 8, 8]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:27, kernel map size:16959
                    [16, 16, 16]->[8, 8, 8]:        gpu_kernel_map: number of unique maps:27, kernel map size:4392
                    [16, 16, 16]->[0, 0, 0]:        gpu_kernel_map: number of unique maps:4, kernel map size:415
                    [8, 8, 8]->[16, 16, 16]:        gpu_kernel_map: number of unique maps:27, kernel map size:4392
                    [8, 8, 8]->[0, 0, 0]:   gpu_kernel_map: number of unique maps:4, kernel map size:1580
                    [4, 4, 4]->[0, 0, 0]:   gpu_kernel_map: number of unique maps:4, kernel map size:5974
                    [1, 1, 1]->[0, 0, 0]:   gpu_kernel_map: number of unique maps:4, kernel map size:18799
                    [16, 16, 16]->[16, 16, 16]:     gpu_kernel_map: number of unique maps:1, kernel map size:415
                    [8, 8, 8]->[8, 8, 8]:   gpu_kernel_map: number of unique maps:1, kernel map size:1580
                    [4, 4, 4]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:13, kernel map size:7112
                    [1, 1, 1]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:27, kernel map size:22245
                    [2, 2, 2]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:22652
                    [16, 16, 16]->[16, 16, 16]:     gpu_kernel_map: number of unique maps:27, kernel map size:5561
                    [8, 8, 8]->[8, 8, 8]:   gpu_kernel_map: number of unique maps:27, kernel map size:22906
                    [4, 4, 4]->[8, 8, 8]:   gpu_kernel_map: number of unique maps:27, kernel map size:16959
                    [1, 1, 1]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:27, kernel map size:52867
                    [1, 1, 1]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:34980
                    [2, 2, 2]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:126294
                    [4, 4, 4]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:27, kernel map size:86736
                    [16, 16, 16]->[16, 16, 16]:     gpu_kernel_map: number of unique maps:1, kernel map size:415
                    [8, 8, 8]->[8, 8, 8]:   gpu_kernel_map: number of unique maps:5, kernel map size:1722
                    [1, 1, 1]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:27, kernel map size:23751
                    [2, 2, 2]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:27336
                    [4, 4, 4]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:27, kernel map size:12116
                    [16, 16, 16]->[16, 16, 16]:     gpu_kernel_map: number of unique maps:7, kernel map size:457
                    [8, 8, 8]->[8, 8, 8]:   gpu_kernel_map: number of unique maps:27, kernel map size:4364
                    [1, 1, 1]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:27, kernel map size:29395
                    [2, 2, 2]->[2, 2, 2]:   gpu_kernel_map: number of unique maps:27, kernel map size:36918
                    [4, 4, 4]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:27, kernel map size:20846
                    [1, 1, 1]->[1, 1, 1]:   gpu_kernel_map: number of unique maps:125, kernel map size:149707
                    [2, 2, 2]->[0, 0, 0]:   gpu_kernel_map: number of unique maps:4, kernel map size:14568
                    [2, 2, 2]->[4, 4, 4]:   gpu_kernel_map: number of unique maps:27, kernel map size:39616
                    algorithm=MinkowskiAlgorithm.DEFAULT
              )
              spatial dimension=3)
  
    target:  tensor([True, True, True,  ..., True, True, True])
    keep:    tensor([True, True, True,  ..., True, True, True])
    
    
    x, pc_data = torch.Size([18799, 1]) torch.Size([150408, 4]) 
    '''
    # _, out_cls, target, keep = model(x, pc_data= pc_data, device=device, prune=False)
    out_cls = model(x)

    # 前向传播后
    print("After forward pass:")
    print_gpu_memory(gpu_index)
    '''   
    out_cls.F.squeeze()：
    tensor([1.2300, 3.3449, 1.3790,  ..., 2.0339, 1.1265, 1.3687], device='cuda:0',
       grad_fn=<SqueezeBackward0>)
    
    target.type(out_cls.F.dtype)：
    tensor([1., 1., 1.,  ..., 1., 1., 1.])
    '''
    loss = crit(out_cls.F.squeeze(), y)
    # metrics = get_metrics(keep, target)
        
    if torch.isnan(loss) or torch.isinf(loss):
        logger.info(f'\n== loss is nan ==, Step: {i}\n')
        continue
        
    # Backward.    
    loss.backward()
    # 反向传播后
    print("After backward pass:")
    print_gpu_memory(gpu_index)

    # Optional clip gradient. 
    if args.max_norm != 0:
      # clip by norm
      max_grad_before = max(p.grad.data.abs().max() for p in model.parameters())
      total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
      
      if total_norm > args.max_norm:

        def get_total_norm(parameters, norm_type=2):
          total_norm = 0.
          for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
          total_norm = total_norm ** (1. / norm_type)
          return total_norm

        print('total_norm:',
          '\nBefore: total_norm:,', total_norm, 
          'max grad:', max_grad_before, 
          '\nthreshold:', args.max_norm, 
          '\nAfter:', get_total_norm(model.parameters()), 
          'max grad:', max(p.grad.data.abs().max() for p in model.parameters()))

    if args.clip_value != 0:
      torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
      print('after gradient clip',  max(p.grad.data.abs().max() for p in model.parameters()))
    
    optimizer.step()
    print("After optimizer step:")
    print_gpu_memory(gpu_index)

    # record.
    # with torch.no_grad():
    #   sum_loss += loss.item()
    #   all_metrics += np.array(metrics)
    
    # Display.
    if i % args.base_step == 0:
      # average.
      with torch.no_grad():
        sum_loss /= args.base_step
        all_metrics /= args.base_step

      if np.isinf(sum_loss):
        logger.info('inf error!')
        sys.exit(0)
        
      if np.isnan(sum_loss):
        logger.info('NaN error!')
        sys.exit(0)

      # logger.
      logger.info(f'\nIteration: {i}')
      logger.info(f'Running time: {((time.time()-start_time)/60):.2f} min')
      logger.info(f'Data Loading time: {dataloader_time:.5f} s')
      logger.info(f'Sum Loss: {sum_loss:.4f}')
      logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')


      # writer.
      writer.add_scalars(main_tag='train/losses', 
                        tag_scalar_dict={'sum_loss' :sum_loss},
                        global_step=i)

      writer.add_scalars(main_tag='train/metrics', 
                        tag_scalar_dict={'Precision': all_metrics[0,0],
                                         'Recall': all_metrics[0,1],
                                         'IoU': all_metrics[0,2]},
                        global_step=i)
      
      # return 0.
      sum_loss = 0.
      all_metrics = np.zeros((1,3))

      # empty cache.
      torch.cuda.empty_cache()
      
    if i % (args.test_step) == 0:
      logger.info(f'\n==========Evaluation: iter {i}==========')
      # save.
      logger.info(f'save checkpoints: {ckptdir}/iter{str(i)}')
      torch.save({'step': i, 'model': model.state_dict(),
                  'optimizer':  optimizer.state_dict(),
                  }, os.path.join(ckptdir, 'iter' + str(i) + '.pth'))

      # Evaluation.
      logger.info(f'\n=====Evaluation: iter {i} =====')
      with torch.no_grad():
        test1(model=model, test_dataloader=test_dataloader, 
          logger=logger, writer=writer, writername='eval', step=i, args=args, device=device)

      torch.cuda.empty_cache()

    # if i % (args.test_step) == 0:
    #   # Evaluation 8i.
    #   logger.info(f'\n=====Evaluation: iter {i} 8i =====')
    #   with torch.no_grad():
    #     test2(model=model, test_dataloader=test_dataloader2,
    #       logger=logger, writer=writer, writername='eval_8i', step=i, test_pc_error=True, args=args, device=device, First_eval=First_eval)
    #     First_eval=False
    #   torch.cuda.empty_cache()
    #
    #   model.to(device)

    if i % int(args.lr_step) == 0:
      scheduler.step()
      logger.info(f'LR: {scheduler.get_lr()}')
    
  writer.close()
    
if __name__ == '__main__':
  args = parse_args()

  logdir = os.path.join(args.logdir, args.prefix)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  if not os.path.exists('tmp'):
    os.makedirs('tmp')
  logger = getlogger(logdir)
  logger.info(args)
  writer = SummaryWriter(log_dir=logdir)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  logger.info(f'Device:{device}')

  # Load data. 所有的shapenet数据都是XXX.h5  123.h5
  filedirs = glob.glob(args.dataset+'*.ply')
  filedirs = sorted(filedirs)
  logger.info(f'1-Train Files(8i) length: {len(filedirs)}')
  # 训练集和测试集
  train_dataloader = make_data_loader(files=filedirs[int(args.num_test):],
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=mp.cpu_count(),
                                      repeat=True)
  
  test_dataloader = make_data_loader( files=filedirs[:int(args.num_test)],
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=mp.cpu_count(),
                                      repeat=False)


  # 8i dataset
  # all 8i ply file
  # eighti_filedirs = glob.glob(args.dataset_8i+'*.ply')
  # # sort
  # eighti_filedirs = sorted(eighti_filedirs)
  # logger.info(f'8I Files length: {len(eighti_filedirs)}')

  # eighti_dataloader = make_data_loader( files=eighti_filedirs,
  #                                       batch_size=1,
  #                                       shuffle=False,
  #                                       num_workers=1,
  #                                       repeat=False)


  # Network.
  model = MyNet().to(device)
  # logger.info(model)

  train(model, train_dataloader, test_dataloader, logger, writer, args, device)

