# example.py
import habitat
import habitat.analysis
pred = habitat.analysis.predictor.Predictor()

for bs in range(1, 65):
    pred_origin = pred.conv2d_pred.predict([0, bs, 112, 64, 64, 3, 1, 1],  'RTX2070')
    pred_target = pred.conv2d_pred.predict([0, bs, 112, 64, 64, 3, 1, 1],  'RTX2080Ti')
    print(bs, pred_origin, pred_target)