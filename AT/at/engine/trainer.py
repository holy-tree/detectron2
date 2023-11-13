import os
import time
import logging
import torch
import copy
import pprint
import cv2
import math
import torch.nn as nn
import itertools

from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage


from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks
from detectron2.engine import DefaultTrainer, TrainerBase, SimpleTrainer
from detectron2.engine.train_loop import AMPTrainer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    DatasetEvaluators, 
    PascalVOCDetectionEvaluator, 
    COCOEvaluator, 
    verify_results
)


import at.modeling.meta_arch.adain as adain
from at.data.dataset_mapper import DatasetMapperTwoCropSeparate
from at.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from at.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from at.solver.build import build_lr_scheduler
from at.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops
)

from .probe import OpenMatchTrainerProbe


from detectron2.structures import ImageList
from typing import Dict, Tuple, List, Optional

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm)
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
            If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
            a `last_checkpoint` file), resume from the file. Resuming means loading all
            available states (eg. optimizer and scheduler) and update iteration counter
            from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
            Otherwise, this is considered as an independent training. The method will load model
            weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
            from iteration 0.
            Args:
                resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    # 无效函数
    def train_loop(self, start_iter: int, max_iter: int):

        print(123)

        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]
        
        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
        
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results
        
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret 

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # adainNet, adain_optimizer1, adain_optimizer2 = self.build_adainNet_opt(cfg)
        # self.adainNet = adainNet
        # self.adain_optimizer1 = adain_optimizer1
        # self.adain_optimizer2 = adain_optimizer2

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        
        pprint.pprint(model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        
        # Ensemble teacher and student model is for model saving and loading
        # !!! new add !!!
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=False):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_adainNet_opt(cls, cfg):
        decoder = adain.decoder
        vgg = adain.vgg
        fc1 = adain.fc1
        fc2 = adain.fc2
       
        vgg.load_state_dict(torch.load(cfg.MODEL.VGG)['model'])
        vgg = nn.Sequential(*list(vgg.children())[:19])

        adainNet = adain.AdainNet(vgg, decoder,fc1,fc2)
        adainNet.train()
        adainNet.to(torch.device('cuda'))

        adain_optimizer1 = torch.optim.Adam(itertools.chain(*[adainNet.dec_1.parameters(),adainNet.dec_2.parameters(), adainNet.dec_3.parameters(), adainNet.dec_4.parameters()]), lr=1e-4)
        adain_optimizer2 = torch.optim.Adam(itertools.chain(*[adainNet.fc1.parameters(),adainNet.fc2.parameters()]), lr=1e-4)

        return adainNet, adain_optimizer1, adain_optimizer2

    @classmethod
    def build_train_loader(cls, cfg):
        # mapper_label = DatasetMapperTwoCropSeparate(cfg, is_train=True, is_label=True)
        # mapper_unlabel = DatasetMapperTwoCropSeparate(cfg, is_train=True, is_label=False)
        mapper = DatasetMapperTwoCropSeparate(cfg, is_train=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        thres_score = []
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres
            
            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres            
            thres_score = proposal_bbox_inst.scores - thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            
            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)
            thres_score = thres_score[valid_map]

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        return new_proposal_inst, thres_score

    
    def process_pseudo_label(self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""):
        list_instances = []
        list_thres_score = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst, thres_score = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
            list_thres_score.append(thres_score)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output, list_thres_score
    
    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        return label_list
    
    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        # !!! new add
        # 取消正则化
        images = [(x["image"]).float().to(torch.device('cuda')) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
        )
        return images



    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        # 由于 IMG_PER_BATCH_LABEL: 2，因此 label_data_q 包含两张img
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        
        # print(label_data_k[0]['instances'].get_fields()['gt_boxes'])
        # print(label_data_k[0]['instances'].get_fields()['gt_boxes'].tensor)
       

        # if not label_data_k[0]['image_id'] == '00247':
        #     return
        # print(label_data_k[0]['image_id'])
        # # print(label_data_k[0]['instances'])
        # image1 = label_data_k[0]['image'].clone().detach().double().to(torch.device('cpu'))
        # image1_np = np.ascontiguousarray(image1.numpy().transpose((1,2,0)))
        # img_box = label_data_k[0]['instances'].get_fields()['gt_boxes'].tensor
        # for box in img_box:
        #     box = box.int()
        #     left = (box[0].item(), box[1].item())
        #     right = (box[2].item(), box[3].item())
        #     image1_np = cv2.rectangle(image1_np, left, right,(0,0,255),5)
        #     # image1_np = cv2.putText(image1_np, 'n', left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 3)
        # cv2.imwrite("./2.jpg", image1_np)
        # quit()

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())


            # !!! new add
            # adain组件部分
            # 更新decoder参数
            # img = self.preprocess_image(label_data_k + unlabel_data_k).tensor
            # content_img = img[:2]
            # style_img = img[2:]
            # print(f"content:{content_img.shape}")
            # print(f"style:{style_img.shape}")


            # loss_c, loss_const = self.adainNet(content_img, style_img, flag=0)
            # loss_c = 1.0 * loss_c
            # loss_const = 1.0 *loss_const
            # loss_cont = loss_c + loss_const

            # self.adain_optimizer1.zero_grad()
            # self.adain_optimizer2.zero_grad()
            # loss_cont.backward()
            # self.adain_optimizer1.step()
            # record_dict['loss_cont'] = loss_cont

            # # 更新fc模块参数
            # loss_s_1, loss_s_2 = self.adainNet(content_img, style_img, flag=1)
            # loss_s_1 = 50.0 * loss_s_1
            # loss_s_2 = 1.0 * loss_s_2
            # loss_style =  loss_s_1 + loss_s_2

            # self.adain_optimizer1.zero_grad()
            # self.adain_optimizer2.zero_grad()
            # loss_style.backward()
            # self.adain_optimizer2.step()
            # record_dict['loss_style'] = loss_style
        
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()
            
            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            
            record_dict = {}

            ######################## For probe #################################
            
            # gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)


            # 获取gtbox
            # gt_box = [unlabel_data['instances'].get('gt_boxes') for unlabel_data in unlabel_data_k]
            

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (_,proposals_rpn_unsup_k,proposals_roih_unsup_k,_,) = self.model_teacher(
                    unlabel_data_k, branch="unsup_data_weak")

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']  
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

             #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            #Process pseudo labels and thresholding
            (pesudo_proposals_rpn_unsup_k,nun_pseudo_bbox_rpn,_) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding")
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _, list_thres_score = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            joint_proposal_dict['proposals_pseudo_roih_thres_score'] = list_thres_score            


            # pprint.pprint(proposals_roih_unsup_k)
            # pprint.pprint(pesudo_proposals_roih_unsup_k)

            

            # proposals_roih_unsup_k_bbox = [proposal.get('gt_boxes') for proposal in pesudo_proposals_roih_unsup_k]
            # proposals_roih_unsup_k_score = [proposal.get('scores') for proposal in pesudo_proposals_roih_unsup_k]
            
            # image1 = unlabel_data_k[0]['image'].clone().detach().double().to(torch.device('cpu'))
            # image2 = unlabel_data_k[1]['image'].clone().detach().double().to(torch.device('cpu'))

            # image1_np = np.ascontiguousarray(image1.numpy().transpose((1,2,0)))
            # image2_np = np.ascontiguousarray(image2.numpy().transpose((1,2,0)))

            
            # img_box, box_score = proposals_roih_unsup_k_bbox[0], proposals_roih_unsup_k_score[0]
            # for box, score, gt_box_0 in zip(img_box, box_score, gt_box[0]):
            #     box = box.int()
            #     gt_box_0 = gt_box_0.int()
            #     left = (box[0].item(), box[1].item())
            #     right = (box[2].item(), box[3].item())
            #     gt_left = (gt_box_0[0].item(), gt_box_0[1].item())
            #     gt_right = (gt_box_0[2].item(), gt_box_0[3].item())
            #     image1_np = cv2.rectangle(image1_np, left, right,(0,255,0),2)
            #     image1_np = cv2.rectangle(image1_np, gt_left, gt_right,(0,0,255),2)
            #     image1_np = cv2.putText(image1_np, '{:.3f}'.format(score), left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # img_box, box_score = proposals_roih_unsup_k_bbox[1], proposals_roih_unsup_k_score[1]
            # for box, score, gt_box_1 in zip(img_box, box_score, gt_box[1]):
            #     box = box.int()
            #     gt_box_1 = gt_box_1.int()
            #     left = (box[0].item(), box[1].item())
            #     right = (box[2].item(), box[3].item())
            #     gt_left = (gt_box_1[0].item(), gt_box_1[1].item())
            #     gt_right = (gt_box_1[2].item(), gt_box_1[3].item())
            #     image2_np = cv2.rectangle(image2_np, left, right,(0,255,0),2)
            #     image2_np = cv2.rectangle(image2_np, gt_left, gt_right,(0,0,255),2)
            #     image2_np = cv2.putText(image2_np, '{:.3f}'.format(score), left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.imwrite("./unlabel_image1.jpg", image1_np)
            # cv2.imwrite("./unlabel_image2.jpg", image2_np)
            # quit()
            

           

            # 3. add pseudo-label to unlabeled data
            unlabel_data_q = self.add_label(unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
            unlabel_data_k = self.add_label(unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q


            # !!! new add   content alignment
            # input strongly and weakly augnmented labeled data into content alignment module
            record_content_loss = self.model(all_label_data, branch="content_alignment")
            record_dict.update(record_content_loss)

            
            
            # 4. input both strongly and weakly augmented labeled data into student model
            # record_all_label_data, _, _, _ = self.model(all_label_data, branch="supervised")
            # record_dict.update(record_all_label_data)

            # !!! new add
            # 输入student模型的数据修改为经过 adain层之后的
            all_data = all_label_data + all_unlabel_data
            record_all_label_data, _, _, _ = self.model(all_data, branch="adain_supervised")
            record_dict.update(record_all_label_data)


            # 5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, _ = self.model(all_unlabel_data, branch="supervised_target")
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]   
            # new add!!!
            # 优化伪标签生成部分，对于精确度高的伪标签赋予更高的权值         
            # thres = sum(joint_proposal_dict['proposals_pseudo_roih_thres_score'][0]) + sum(joint_proposal_dict['proposals_pseudo_roih_thres_score'][1])            
            # for key in new_record_all_unlabel_data:
            #     new_record_all_unlabel_data[key] *= math.exp(thres)
            record_dict.update(new_record_all_unlabel_data)
            
            
            # !!! new add
            # cancel the dis_loss
            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            # for i_index in range(len(unlabel_data_k)):
            #     # unlabel_data_item = {}
            #     for k, v in unlabel_data_k[i_index].items():
            #         # label_data_k[i_index][k + "_unlabeled"] = v
            #         label_data_k[i_index][k + "_unlabeled"] = v
            #     # unlabel_data_k[i_index] = unlabel_data_item
            # all_domain_data = label_data_k
            # # all_domain_data = label_data_k + unlabel_data_k
            # record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
            # record_dict.update(record_all_domain_data)


            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT)
                    elif (key == "loss_D_img_s" or key == "loss_D_img_t"):  # set weight for discriminator
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                    elif key == "loss_cont_cons":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CONT_CONS_LOSS_WEIGHT
                    elif key == "loss_ins_cons":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_CONS_LOSS_WEIGHT
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1


            losses = sum(loss_dict.values())


        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (student_model_dict[key] * (1 - keep_rate) + value * keep_rate)
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)
    
    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
    

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret