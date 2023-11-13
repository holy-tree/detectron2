import torch
import os

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch, default_setup

from at.config import add_at_config
from at.engine.trainer import ATeacherTrainer, BaselineTrainer
from at.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import at.data.register

from at.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from at.modeling.proposal_generator.rpn import PseudoLabRPN
from at.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup(args):

    cfg = get_cfg()
    add_at_config(cfg)
    # args.config_file = "./configs/faster_rcnn_R101_cross_tn->h60_1500.yaml"
    # args.config_file = "./configs/TN->H60_FPN_pretrain.yaml"
    args.config_file = "./configs/test.yaml"
    # args.config_file = "./configs/H60.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.WEIGHTS="./weights/TN_FPN.pth"

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found!")
    
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            # adainNet, _,_ = Trainer.build_adainNet_opt(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)

    torch.cuda.set_device(0)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


    

