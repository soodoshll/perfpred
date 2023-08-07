# example.py
import habitat
import habitat.analysis
pred = habitat.analysis.predictor.Predictor()

image_size = 224
in_channels = 3
out_channels = 64
kernel_size = 7
stride = 2
padding = 3

for bs in range(1, 65):
    pred_origin = pred.conv2d_pred.predict([0, bs, image_size, in_channels, out_channels, kernel_size, stride, padding],  'RTX2070')
    pred_target = pred.conv2d_pred.predict([0, bs, image_size, in_channels, out_channels, kernel_size, stride, padding],  'RTX3090')
    print(bs, pred_origin, pred_target)
