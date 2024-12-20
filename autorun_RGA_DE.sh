# echo baseline
# python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res56_res20.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res110_res32.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/wrn40_2_wrn_16_2.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/wrn40_2_wrn_40_1.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/different_style/res32x4_shuv1.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD/different_style/res32x4_shuv2.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"

# echo RGA
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/res56_res20.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/res110_res32.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/wrn40_2_wrn_16_2.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/wrn40_2_wrn_40_1.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/same_style/vgg13_vgg8.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/different_style/res32x4_shuv1.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"
# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA/different_style/res32x4_shuv2.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 EXPERIMENT.PROJECT "yun_1_cifar100_128_005"

# python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/res32x4_res8x4.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" EXPERIMENT.TAG "CAT_FUSION" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/res56_res20.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/res110_res32.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/wrn40_2_wrn_16_2.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/wrn40_2_wrn_40_1.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/same_style/vgg13_vgg8.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/different_style/res32x4_shuv1.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 
python tools/train.py --cfg configs/cifar100/ReviewKD_RGA-decoupled/different_style/res32x4_shuv2.yaml EXPERIMENT.PROJECT "yun_1_cifar100_128_005" SOLVER.BATCH_SIZE 128 SOLVER.LR 0.05 