import sys
import shutil

sys.path.append('.')

import os
import cv2
import tqdm
import json
import numpy
import wandb
import torch
import argparse
import torch_redstone as rst
from sklearn.cluster import KMeans
from lib.models.autoencoders import MultiSceneNeRF
from mmgen.models import build_model, build_module
from lib.core.optimizer import build_optimizers
from lib.core.ssdnerf_gui import OrbitCamera
from lib.datasets.shapenet_srn import ShapeNetSRN
from lib.datasets.nerf_synthetic import NerfSynthetic
from lib.datasets.dtu import DTUDataset
from PIL import Image
import einops


torch.backends.cuda.matmul.allow_tf32 = True

def fps_downsample(points, n_points_to_sample):
    selected_points = numpy.zeros((n_points_to_sample, 3))
    selected_idxs = []
    dist = numpy.ones(points.shape[0]) * 100
    for i in range(n_points_to_sample):
        idx = numpy.argmax(dist)
        selected_points[i] = points[idx]
        selected_idxs.append(idx)
        dist_ = ((points - selected_points[i]) ** 2).sum(-1)
        dist = numpy.minimum(dist, dist_)

    return selected_idxs

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()


# obj = sys.argv[-1]  # 'bonsai'
argp = argparse.ArgumentParser()
obj = 'chair'
img = sys.argv[-1]
img_name = os.path.splitext(os.path.basename(img))[0]
dynamic = False
model_res = 20
# world_scale = 0.2
# model_res = 128
model_scaling_factor = 16
# model_scaling_factor = 1
model_ch = 8
# model_ch = 256
n_rays = 2 ** 12
code_lr = 0.0
net_lr = 0.002
seed = 1337
n_views = 6
n_val = 1
learn_bg = False
load_image = img
print(load_image)
net_lr_decay_to = 0.002 if load_image else 0.001
n_iters = 5000 if load_image else 10000
val_iter = 1000
bg_init = 1.0
wandb_project = "zerorf"
device = 'cuda:0'
BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)

code_size = (3, model_ch, model_res, model_res)
proj_name = "zerorf-%s-dif" % (img_name.replace("_", "-"))
# proj_name = "%s-ten4d-20x" % (obj.replace("_", "-"))
# code_size = (model_ch, model_res, model_res, model_res)

rst.seed(seed)

poses = []
intrinsics = []
if load_image:
    image = numpy.array(Image.open(load_image)).astype(numpy.float32) / 255.0
    image = torch.tensor(image).cuda()
    images = einops.rearrange(image, '(ph h) (pw w) c -> (ph pw) h w c', ph=3, pw=2)[None]
    meta = json.load(open(os.path.join(os.path.dirname(__file__), "meta.json")))
    poses = numpy.array([
        (numpy.array(frame['transform_matrix']) @ BLENDER_TO_OPENCV_MATRIX) * 2
        for frame in meta['sample_0']['view_frames']
    ])
    _, b, h, w, c = images.shape
    x, y = w / 2, h / 2
    focal_length = y / numpy.tan(meta['fovy'] / 2)
    intrinsics = numpy.array([[focal_length, focal_length, x, y]] * n_views)

model_scale = dict(chair=2.1, drums=2.3, ficus=2.3, hotdog=3.0, lego=2.4, materials=2.4, mic=2.5, ship=2.75)
if os.name == "nt":
    basedir = "I:/UCSD/ssdnerf-data/nerf_synthetic"
else:
    basedir = "/root/nerf_synthetic"
    # basedir = "/host/home/r5shi/data_DTU"
world_scale = (2 / model_scale[obj] if not dynamic else 0.666)
# dataset = DTUDataset(f"{basedir}/{obj}", split='train', world_scale=world_scale, rgba=True)
# val = DTUDataset(f"{basedir}/{obj}", split='test', world_scale=world_scale)
# test = DTUDataset(f"{basedir}/{obj}", split='test', world_scale=world_scale)
dataset = NerfSynthetic([f"{basedir}/{obj}/transforms_train.json"], rgba=True, world_scale=world_scale)
val = NerfSynthetic([f"{basedir}/{obj}/transforms_val.json"], world_scale=world_scale)
test = NerfSynthetic([f"{basedir}/{obj}/transforms_test.json"], world_scale=world_scale)
work_dir = "results/%s" % proj_name
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

entry = dataset[0]
# selected_idxs = dict(
#     ship=[50,17,9,31],
#     mic=[52,37,73,64],
#     materials=[33,39,36,5],
#     hotdog=[22,62,96,23],
#     ficus=[23,52,24,91],
#     drums=[80,65,72,66],
#     chair=[55,87,94,5],
#     lego=[16,57,28,1],
# )[obj]
selected_idxs = kmeans_downsample(entry['cond_poses'][..., :3, 3], n_views)
# selected_idxs = list(range(len(entry['cond_imgs'])))
data_entry = dict(
    cond_imgs=torch.tensor(entry['cond_imgs'][selected_idxs][None]).float().to(device),
    cond_poses=torch.tensor(entry['cond_poses'])[selected_idxs][None].float().to(device),
    cond_intrinsics=torch.tensor(entry['cond_intrinsics'])[selected_idxs][None].float().to(device),
    cond_times=torch.tensor(entry['cond_times'])[selected_idxs][None].float().to(device) if dynamic else None,
    scene_id=[0],
    scene_name=[proj_name]
)
entry = val[0]
val_entry = dict(
    test_imgs=torch.tensor(entry['cond_imgs'][:n_val][None]).float().to(device),
    test_poses=torch.tensor(entry['cond_poses'][:n_val])[None].float().to(device),
    test_intrinsics=torch.tensor(entry['cond_intrinsics'][:n_val])[None].float().to(device),
    test_times=torch.tensor(entry['cond_times'])[:n_val][None].float().to(device) if dynamic else None,
    scene_id=[0],
    scene_name=[proj_name]
)
entry = test[0]
test_entry = dict(
    test_imgs=torch.tensor(entry['cond_imgs'][:][None]).float().to(device),
    test_poses=torch.tensor(entry['cond_poses'][:])[None].float().to(device),
    test_intrinsics=torch.tensor(entry['cond_intrinsics'][:])[None].float().to(device),
    test_times=torch.tensor(entry['cond_times'])[:][None].float().to(device) if dynamic else None,
    scene_id=[0],
    scene_name=[proj_name]
)
if load_image:
    data_entry = dict(
        cond_imgs=images,
        cond_poses=torch.tensor(poses)[None].float().to(device) * 0.9,
        cond_intrinsics=torch.tensor(intrinsics)[None].float().to(device),
        scene_id=[0],
        scene_name=[proj_name]
    )

pic_h = data_entry['cond_imgs'].shape[-3]
pic_w = data_entry['cond_imgs'].shape[-2]
if load_image:
    model_res = 4
    pic_h = pic_w = 320
cam = OrbitCamera('render', pic_w, pic_h, 3.2, 48)

decoder_1 = dict(
    type='TensorialDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=model_ch, out_ch=16, noise_res=model_res,
        tensor_config=(
            ['xy', 'yz', 'zx'] if load_image else
            ['xy', 'z', 'yz', 'x', 'zx', 'y'] if not dynamic else
            ['xy', 'xt', 'yt', 'yz', 'yt', 'zt', 'xz', 'xt', 'zt']
        )
    ),
    subreduce=1 if load_image else 2 if not dynamic else 3,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    max_steps=1024 if not load_image else 320,
    n_images=n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=dynamic,
    visualize_mesh=True
)
decoder_2 = dict(
    type='FreqFactorizedDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=model_ch, out_ch=16, noise_res=model_res,
        tensor_config=['xyz', 'xyz']
    ),
    subreduce=1,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    max_steps=1024 if not load_image else 640,
    n_images=n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=dynamic,
    freq_bands=[None, 0.4],
    visualize_mesh=True
)

patch_reg_loss = build_module(dict(
    type='MaskedTVLoss',
    power=1.5,
    loss_weight=0.00
))
nerf: MultiSceneNeRF = build_model(dict(
    type='MultiSceneNeRF',
    code_size=code_size,
    code_activation=dict(type='IdentityCode'),
    grid_size=64,
    patch_size=32,
    decoder=decoder_2,
    decoder_use_ema=False,
    bg_color=1.0,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=3.2
    ),
    use_lpips_metric=torch.cuda.mem_get_info()[1] // 1000 ** 3 >= 32,
    cache_size=1,
    cache_16bit=False,
    init_from_mean=True
), train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.05,
    extra_scene_step=8 if code_lr > 0 else 0,
    n_inverse_rays=n_rays,
    n_decoder_rays=n_rays,
    loss_coef=0.1 / (pic_h * pic_w),
    optimizer=dict(type='Adam', lr=code_lr, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.99),
    cache_load_from=None,
    viz_dir=None,
    loss_denom=1.0,
    decoder_grad_clip=1.0
),
test_cfg = dict(
    img_size=(pic_h, pic_w),
    density_thresh=0.01,
    max_render_rays=pic_h * pic_w,
    dt_gamma_scale=0.5,
    n_inverse_rays=n_rays,
    loss_coef=0.1 / (pic_h * pic_w),
    n_inverse_steps=400,
    optimizer=dict(type='Adam', lr=0.0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998),
    return_depth=False
))

nerf.bg_color = nerf.decoder.bg_color = torch.nn.Parameter(torch.ones(3) * bg_init, requires_grad=learn_bg)
nerf.to(device)
nerf.train()
optim = build_optimizers(nerf, dict(decoder=dict(type='AdamW', lr=net_lr, foreach=True, weight_decay=0.2, betas=(0.9, 0.98))))
# lr_sched = get_scheduler('cosine', optim['decoder'], num_warmup_steps=0, num_training_steps=n_iters)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim['decoder'], n_iters, eta_min=net_lr_decay_to)

wandb.init(
    project=wandb_project,
    name=proj_name,
    save_code=True,
    config=dict(selected_idxs=selected_idxs)
)
prog = tqdm.trange(n_iters)
best_psnr = 0.0

for j in prog:
    lv = nerf.train_step(data_entry, optim)['log_vars']
    lr_sched.step()
    lv.pop('code_rms')
    lv.pop('loss')
    prog.set_postfix(**lv)
    wandb.log(dict(train=lv))
    if j == 50:
        nerf.train_cfg['n_inverse_rays'] = 2 ** 14
        nerf.train_cfg['n_decoder_rays'] = 2 ** 14
    if j == 100:
        # nerf.patch_reg_loss = patch_reg_loss
        nerf.train_cfg['n_inverse_rays'] = 2 ** 16 if load_image else 2 ** 17
        nerf.train_cfg['n_decoder_rays'] = 2 ** 16 if load_image else 2 ** 17
    if j % val_iter == val_iter - 1:
        cam = OrbitCamera('final', pic_w, pic_h, 3.2, 48)
        cache = nerf.cache[0]
        nerf.eval()
        if not load_image:
            with torch.no_grad():
                if os.path.exists("viz"):
                    shutil.rmtree("viz")
                log_vars, _ = nerf.eval_and_viz(
                    val_entry, nerf.decoder,
                    cache['param']['code_'][None].to(device),
                    cache['param']['density_bitfield'][None].to(device),
                    "viz",
                    cfg=nerf.test_cfg
                )
            print()
            print(log_vars)
            wandb.log(dict(val=log_vars))
            this_psnr = log_vars['test_psnr']
        if load_image or this_psnr >= best_psnr or j == len(prog) - 1:
            torch.save(nerf.state_dict(), open("nerf-zerorf.pt", "wb"))
            best_psnr = this_psnr if not load_image else 0
            out = cv2.VideoWriter('dec_%d.avi' % j, cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (pic_w, pic_h))
            with torch.no_grad():
                for i in tqdm.trange(60, desc='%.2f' % best_psnr):
                    test_pose = cam.pose
                    test_intrinsic = cam.intrinsics
                    test_time = i / 60 * 2 - 1
                    render_result = nerf.render(
                        nerf.decoder,
                        cache['param']['code_'][None].to(device),
                        cache['param']['density_bitfield'][None].to(device),
                        pic_h, pic_w,
                        torch.tensor(test_intrinsic[None, None]).float().to(device),
                        torch.tensor(test_pose[None, None]).float().to(device),
                        torch.tensor(test_time).reshape(1, 1).float().to(device) if dynamic else None,
                        nerf.test_cfg
                    )
                    cam.orbit(60, numpy.sin(i / 60 * numpy.pi * 2) * 24)
                    frame = render_result[0].squeeze().float().cpu()
                    if not numpy.isfinite(frame).all():
                        print("Non-finite value!")
                    out.write(cv2.cvtColor(
                        (torch.clamp(frame, 0, 1).numpy() * 255).astype(numpy.uint8),
                        cv2.COLOR_RGB2BGR
                    ))
                if j == len(prog) - 1:
                    log_vars, _ = nerf.eval_and_viz(
                        dict(
                            test_poses=data_entry['cond_poses'],
                            test_intrinsics=data_entry['cond_intrinsics'],
                            test_times=data_entry.get('cond_times'),
                            scene_id=[0],
                            scene_name=["0"]
                        ), nerf.decoder,
                        cache['param']['code_'][None].to(device),
                        cache['param']['density_bitfield'][None].to(device),
                        "viz/train_viz",
                        cfg=nerf.test_cfg
                    )
                    wandb.log(dict(train_final=log_vars))
                    if not load_image:
                        log_vars, _ = nerf.eval_and_viz(
                            test_entry, nerf.decoder,
                            cache['param']['code_'][None].to(device),
                            cache['param']['density_bitfield'][None].to(device),
                            "viz/test_viz",
                            cfg=nerf.test_cfg
                        )
                        print()
                        print('Final test:')
                        print(log_vars)
                        wandb.log(dict(test=log_vars))
            out.release()
