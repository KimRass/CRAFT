def get_confidence_score_map(gt_length, pred_length):
    conf_score_map = (
        gt_length - min(gt_length, abs(gt_length - pred_length))
    ) / gt_length
    return conf_score_map
