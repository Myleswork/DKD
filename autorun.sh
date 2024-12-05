python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res32x4_res8x4.yaml
echo "Finished training res32x4_res8x4"

python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res56_res20.yaml
echo "Finished training res32x4_res8x4"

python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/res110_res32.yaml
echo "Finished training res56_res20"

python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/wrn40_2_wrn_16_2.yaml
echo "Finished training wrn40_2_wrn_16_2"

python tools/train.py --cfg configs/cifar100/ReviewKD/same_style/wrn40_2_wrn_40_1.yaml
echo "Finished training wrn40_2_wrn_40_1"
