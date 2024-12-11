echo baseline
# python ../tools/train.py --cfg ../configs/cifar100/ReviewKD/same_style/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1 EXPERIMENT.PROJECT "titanxp_1_cifar100_test" EXPERIMENT.NAME "res32x4_res8x4_128_01"
python ../tools/train.py --cfg ../configs/cifar100/ReviewKD/same_style/vgg13_vgg8.yaml

echo RGA
# python ../tools/train.py --cfg ../configs/cifar100/ReviewKD/same_style/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1 EXPERIMENT.PROJECT "titanxp_1_cifar100_test" EXPERIMENT.NAME "res32x4_res8x4_128_01"
python ../tools/train.py --cfg ../configs/cifar100/ReviewKD_RGA/same_style/vgg13_vgg8.yaml