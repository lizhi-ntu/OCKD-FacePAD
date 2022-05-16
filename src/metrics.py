import numpy as np

def HTER(pred=None, label=None, threshold=None):
    label[label>1]=1
    pred_label = (pred > threshold)
    
    GN = (label == 0).sum() 
    FP = ((label == 0) & (pred_label == 1)).sum()  
    
    GP = (label == 1).sum()                        
    FN = ((label == 1) & (pred_label == 0)).sum() 
    
    far = FN.item() / GP.item()
    frr = FP.item() / GN.item()
    hter = (far + frr) / 2
    return far, frr, hter

def BPCER(pred=None, label=None, threshold=None):
    pred_label = (pred > threshold)

    GN = (label == 0).sum()
    FP = ((label == 0) & (pred_label == 1)).sum()

    bpcer = FP.item() / GN.item()
    
    return bpcer

def OPT_HTER(pred=None, label=None):
    thresholds = np.unique(pred)
    hter_opt = 1
    left_bound = 0
    right_bound = 0

    for i in range(len(thresholds)):
        threshold = thresholds[i]
        far, frr, hter = HTER(pred, label, threshold)

        if hter < hter_opt:
            left_bound = threshold
            right_bound = threshold
            hter_opt = hter
            far_opt, frr_opt, hter_opt = far, frr, hter
        
        if hter == hter_opt:
            right_bound = threshold
    
    threshold_opt = 0.5 * (left_bound + right_bound)    
    return far_opt, frr_opt, hter_opt, threshold_opt

def BPCER10(pred=None, label=None):
    thresholds = np.unique(pred)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        bpcer = BPCER(pred, label, threshold) 
        if bpcer < 0.1:
            threshold10 = threshold
            bpcer10 = bpcer
            break
    
    return bpcer10, threshold10

def EER(pred=None, label=None):
    thresholds = np.unique(pred)
    current_diff = 1
    left_bound = 0                   
    right_bound = 0                  
    
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        far, frr, hter = HTER(pred, label, threshold)
        diff = abs(far - frr)
        if diff < current_diff:
            right_bound = threshold
            left_bound = threshold
            current_diff = diff
            far_eer, frr_eer, eer = far, frr, hter
        if diff == current_diff:
            right_bound = threshold
    threshold_eer = 0.5 * (left_bound + right_bound)    
    
    return far_eer, frr_eer, eer, threshold_eer

