import argparse
import datetime
import logging
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import sys

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco as coco
from model.losses import CTCLoss_neg, get_masked_ptc_loss, get_seg_loss, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils.jigsaw import shuffle_blocks, unshuffle_blocks
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label, cam_to_roi_mask2, crop_from_roi_neg, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("/home/ttmdawn/fjy_yan2/ToCo_yan2/toco/pretrained/")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--img_folder", default='/home/ttmdawn/fangjingyuan/data/COCO2014/JPEGImages/', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/home/ttmdawn/fangjingyuan/data/COCO2014/SegmentationClass/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work_dir_coco_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--spg", default=8, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=500, type=int, help=" logging iters") # 200
parser.add_argument("--eval_iters", default=4000, type=int, help="validation iters") #4000
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.65, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=[1.0,0.5,0.75,1.25,1.5], help="multi_scales for cam")

parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, help="save_ckpt")

parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)
            cls, segs, _, _ = model(inputs,)
            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, num_classes=args.num_classes)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=coco.class_list)

    return cls_score, tab_results

def train(args=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5679'
    dist.init_process_group(backend=args.backend, init_method='env://', rank=0, world_size=1)
    torch.cuda.set_device(args.local_rank)
    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = coco.CocoClsDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=9
    )
    param_groups = model.get_param_groups()
    model.to(device)

    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    def cos_sim(x1, x2):
        b, n1, c = x1.shape
        b, n2, c = x2.shape
        x1 = F.normalize(x1, p=2, dim=2, eps=1e-8)
        x2 = F.normalize(x2, p=2, dim=2, eps=1e-8)
        cos_sim = torch.matmul(x1, x2.transpose(1, 2).contiguous())
        return torch.abs(cos_sim)

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs_jig, jigsaw_seed = shuffle_blocks(inputs)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)
        inputs_fuse = torch.cat((inputs, inputs_jig.cuda()), dim=0)


        # get local crops from uncertain regions
        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                    bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                    ignore_index=args.ignore_index)
        valid_aux_cam, _ = cam_to_label(cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                    bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                    ignore_index=args.ignore_index)
        refined_pseudo_label_aux = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_aux_cam, cls_labels=cls_label,
                                                       high_thre=args.high_thre, low_thre=args.low_thre,
                                                       ignore_index=args.ignore_index, img_box=img_box, )
        if n_iter <= 12000:
            refined_pseudo_label = refined_pseudo_label_aux
        else:
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                           high_thre=args.high_thre, low_thre=args.low_thre,
                                                           ignore_index=args.ignore_index, img_box=img_box, )

        bg_mask = torch.zeros_like(refined_pseudo_label)
        bg_mask[refined_pseudo_label == 0] = 1
        bg_mask = bg_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        bg_img = bg_mask * inputs

        patch1 = bg_img[:, :, 0:224, 0:224]  # 左上角
        patch2 = bg_img[:, :, 0:224, 224:448]  # 右上角
        patch3 = bg_img[:, :, 224:448, 0:224]  # 左下角
        patch4 = bg_img[:, :, 224:448, 224:448]

        crops_bg = torch.stack([patch1, patch2, patch3, patch4], dim=1)

        cls, segs, fmap, cls_aux, pro_pred, mix_feat, cls_token, bg_token, seg_jig = model(inputs_fuse, istrain=True, n_iter=n_iter, bg_img=crops_bg)

        # cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # seg_loss & reg_loss
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss1 = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)
        seg_jig = F.interpolate(seg_jig, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        pseudo_label_jig, _ = shuffle_blocks(refined_pseudo_label.unsqueeze(1), jigsaw_seed)
        pseudo_label_jig = pseudo_label_jig.squeeze(1)
        seg_loss2 = get_seg_loss(seg_jig, pseudo_label_jig.type(torch.long), ignore_index=args.ignore_index)
        seg_loss = seg_loss1 + seg_loss2

        # ptc loss
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # prototype loss
        pro_pred = pro_pred / (F.adaptive_max_pool2d(pro_pred, (1, 1)) + 1e-5)
        if n_iter <=16000:
            pro_loss = get_seg_loss(pro_pred, refined_pseudo_label_aux.type(torch.long), ignore_index=args.ignore_index)
        else:
            pro_loss = get_seg_loss(pro_pred, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        # class pro contrast
        cls_mask = cls_label.unsqueeze(1).cuda()
        bg_pro = mix_feat[:, 0, :].unsqueeze(1)
        fg_pro = mix_feat[:, 1:, :]
        cls_token = cls_token.unsqueeze(1)
        pos_smi = cos_sim(fg_pro, cls_token).transpose(1,2).contiguous()
        neg_smi = cos_sim(bg_pro, cls_token)
        pos_smi_bg = cos_sim(bg_pro, bg_token)
        bg_smi_mask = torch.ones_like(pos_smi_bg)
        b = cls_mask.size(0)
        cpc_loss = 0.25*(1-torch.sum(pos_smi * cls_mask)) / cls_mask.sum() + 0.25*(1-torch.sum(pos_smi_bg * bg_smi_mask)) / bg_smi_mask.sum() + 0.5*torch.sum(neg_smi) / b


        # warmup
        if n_iter <= 8000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * pro_loss + 0.0 * cpc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 12000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.0 * pro_loss + 0.0 * cpc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 16000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.2 * pro_loss + 0.5 * cpc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.2 * pro_loss + 0.5 * cpc_loss + 0.12 * seg_loss + 0.05 * reg_loss


        cls_pred = (cls > 0).type(torch.int16)
        cls_score = np.float64(evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0]))

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'pro_loss': pro_loss.item(),
            'cpc_loss': cpc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
        })

        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, pro_loss: %.4f, cpc_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('pro_loss'), avg_meter.pop('cpc_loss'), avg_meter.pop('seg_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
    # val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
    # if args.local_rank == 0:
    #     logging.info("val cls score: %.6f" % (val_cls_score))
    #     logging.info("\n"+tab_results)
    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
    if dist.is_initialized():
        dist.destroy_process_group()
