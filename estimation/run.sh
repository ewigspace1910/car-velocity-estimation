python speed-estimation.py \
--weights ../_data/model/yolov7.pt \
--classes 2 \
--track_high_thresh 0.3 --track_low_thresh 0.05 --match_thresh 0.7 \
--mode 2 --cfg ../_data/cfg.yaml \
#--save --project "../_data/run/" --name "homography" \
# --device cpu\