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
import torch_redstone as rst
from sklearn.cluster import KMeans
from lib.models.autoencoders import MultiSceneNeRF
from mmgen.models import build_model, build_module
from lib.core.optimizer import build_optimizers
from lib.core.ssdnerf_gui import OrbitCamera
from lib.datasets.nerf_synthetic import NerfSynthetic
from lib.datasets.oppo import OppoDataset
from PIL import Image
import einops
from opt import config_parser
from pprint import pprint

torch.backends.cuda.matmul.allow_tf32 = True

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

args = config_parser()
pprint(args)

model_scaling_factor = 16
device = args.device

BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)

code_size = (3, args.model_ch, args.model_res, args.model_res)

rst.seed(args.seed)

poses = []
intrinsics = []
if args.load_image:
    image = numpy.array(Image.open(args.load_image)).astype(numpy.float32) / 255.0
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
    intrinsics = numpy.array([[focal_length, focal_length, x, y]] * args.n_views)

work_dir = "results/%s" % args.proj_name
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

if not args.load_image:
    if args.dataset == "nerf_syn":
        model_scale = dict(chair=2.1, drums=2.3, ficus=2.3, hotdog=3.0, lego=2.4, materials=2.4, mic=2.5, ship=2.75)
        world_scale = 2 / model_scale[args.obj]
        dataset = NerfSynthetic([f"{args.data_dir}/{args.obj}/transforms_train.json"], rgba=True, world_scale=world_scale)
        val = NerfSynthetic([f"{args.data_dir}/{args.obj}/transforms_val.json"], world_scale=world_scale)
        test = NerfSynthetic([f"{args.data_dir}/{args.obj}/transforms_test.json"], world_scale=world_scale)
        
        entry = dataset[0]
        selected_idxs = kmeans_downsample(entry['cond_poses'][..., :3, 3], args.n_views)
    elif args.dataset == "oi":
        world_scale = 5.0
        dataset = OppoDataset(f"{args.data_dir}/{args.obj}/output", split='train', world_scale=world_scale, rgba=True)
        val = OppoDataset(f"{args.data_dir}/{args.obj}/output", split='test', world_scale=world_scale)
        test = OppoDataset(f"{args.data_dir}/{args.obj}/output", split='test', world_scale=world_scale) 

        entry = dataset[0]
        if args.n_views == 6:
            selected_idxs = [10, 3, 19, 22, 17, 35]
        elif args.n_views == 4:
            selected_idxs = [10, 33, 35, 6]
        else:
            selected_idxs = kmeans_downsample(entry['cond_poses'][..., :3, 3], args.n_views)
    
    data_entry = dict(
        cond_imgs=torch.tensor(entry['cond_imgs'][selected_idxs][None]).float().to(device),
        cond_poses=torch.tensor(entry['cond_poses'])[selected_idxs][None].float().to(device),
        cond_intrinsics=torch.tensor(entry['cond_intrinsics'])[selected_idxs][None].float().to(device),
        scene_id=[0],
        scene_name=[args.proj_name]
    )
    entry = val[0]
    val_entry = dict(
        test_imgs=torch.tensor(entry['cond_imgs'][:args.n_val][None]).float().to(device),
        test_poses=torch.tensor(entry['cond_poses'][:args.n_val])[None].float().to(device),
        test_intrinsics=torch.tensor(entry['cond_intrinsics'][:args.n_val])[None].float().to(device),
        scene_id=[0],
        scene_name=[args.proj_name]
    )
    entry = test[0]
    test_entry = dict(
        test_imgs=torch.tensor(entry['cond_imgs'][:][None]).float().to(device),
        test_poses=torch.tensor(entry['cond_poses'][:])[None].float().to(device),
        test_intrinsics=torch.tensor(entry['cond_intrinsics'][:])[None].float().to(device),
        scene_id=[0],
        scene_name=[args.proj_name]
    )
else:
    data_entry = dict(
        cond_imgs=images,
        cond_poses=torch.tensor(poses)[None].float().to(device) * 0.9,
        cond_intrinsics=torch.tensor(intrinsics)[None].float().to(device),
        scene_id=[0],
        scene_name=[args.proj_name]
    )
    selected_idxs = list(range(args.n_views))

pic_h = data_entry['cond_imgs'].shape[-3]
pic_w = data_entry['cond_imgs'].shape[-2]
if args.load_image:
    args.model_res = 4
    pic_h = pic_w = 320
cam = OrbitCamera('render', pic_w, pic_h, 3.2, 48)

decoder_1 = dict(
    type='TensorialDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch, out_ch=16, noise_res=args.model_res,
        tensor_config=(
            ['xy', 'z', 'yz', 'x', 'zx', 'y']
        )
    ),
    subreduce=1 if args.load_image else 2,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    max_steps=1024 if not args.load_image else 320,
    n_images=args.n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=False,
    visualize_mesh=True
)
decoder_2 = dict(
    type='FreqFactorizedDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch, out_ch=16, noise_res=args.model_res,
        tensor_config=['xyz', 'xyz']
    ),
    subreduce=1,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    max_steps=1024 if not args.load_image else 640,
    n_images=args.n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=False,
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
    decoder=decoder_2 if args.rep == 'dif' else decoder_1,
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
    extra_scene_step=0,
    n_inverse_rays=args.n_rays_init,
    n_decoder_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    optimizer=dict(type='Adam', lr=0, weight_decay=0.),
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
    n_inverse_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    n_inverse_steps=400,
    optimizer=dict(type='Adam', lr=0.0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998),
    return_depth=False
))

nerf.bg_color = nerf.decoder.bg_color = torch.nn.Parameter(torch.ones(3) * args.bg_color, requires_grad=args.learn_bg)
nerf.to(device)
nerf.train()
optim = build_optimizers(nerf, dict(decoder=dict(type='AdamW', lr=args.net_lr, foreach=True, weight_decay=0.2, betas=(0.9, 0.98))))
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim['decoder'], args.n_iters, eta_min=args.net_lr_decay_to)

wandb.init(
    project=args.wandb_project,
    name=args.proj_name,
    save_code=True,
    config=dict(selected_idxs=selected_idxs)
)
prog = tqdm.trange(args.n_iters)
best_psnr = 0.0

for j in prog:
    lv = nerf.train_step(data_entry, optim)['log_vars']
    lr_sched.step()
    lv.pop('code_rms')
    lv.pop('loss')
    prog.set_postfix(**lv)
    wandb.log(dict(train=lv))
    if j == 50:
        nerf.train_cfg['n_inverse_rays'] = round((args.n_rays_init * args.n_rays_up) ** 0.5)
        nerf.train_cfg['n_decoder_rays'] = round((args.n_rays_init * args.n_rays_up) ** 0.5)
    if j == 100:
        nerf.train_cfg['n_inverse_rays'] = args.n_rays_up
        nerf.train_cfg['n_decoder_rays'] = args.n_rays_up
    if j % args.val_iter == args.val_iter - 1:
        cam = OrbitCamera('final', pic_w, pic_h, 3.2, 48)
        cache = nerf.cache[0]
        nerf.eval()
        if not args.load_image:
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
        if args.load_image or this_psnr >= best_psnr or j == len(prog) - 1:
            torch.save(nerf.state_dict(), open("nerf-zerorf.pt", "wb"))
            best_psnr = this_psnr if not args.load_image else 0
            out = cv2.VideoWriter('dec_%d.avi' % j, cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (pic_w, pic_h))
            with torch.no_grad():
                for i in tqdm.trange(60, desc='%.2f' % best_psnr):
                    test_pose = cam.pose
                    test_intrinsic = cam.intrinsics
                    if args.dataset == "oi":
                        revert_y = numpy.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        test_pose = test_pose @ revert_y
                    test_time = i / 60 * 2 - 1
                    render_result = nerf.render(
                        nerf.decoder,
                        cache['param']['code_'][None].to(device),
                        cache['param']['density_bitfield'][None].to(device),
                        pic_h, pic_w,
                        torch.tensor(test_intrinsic[None, None]).float().to(device),
                        torch.tensor(test_pose[None, None]).float().to(device),
                        None,
                        nerf.test_cfg
                    )
                    if args.dataset == "oi":
                        cam.orbit(60, -numpy.sin(i / 60 * numpy.pi * 2) * 24)
                    else:
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
                    if not args.load_image:
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
