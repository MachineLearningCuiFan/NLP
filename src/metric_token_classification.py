# all by cuifan
# 计算每个token二分类的评估结果

def token_evaluation_batch(preds,truth):
    TP = 0
    FP = 0
    FN = 0
    for p_sen,t_sen in zip(preds,truth):
        for p,t in zip(p_sen,t_sen):
            if t == -100:
                continue
            elif p == 1 and  t == 0:
                FP += 1
            elif p == 1 and t == 1:
                TP += 1
            elif p == 0 and t == 1:
                FN += 1
            else:
                continue
    return TP,FP,FN


def compute_prf(TP, FP, FN):
    results = {}
    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0
    results["P"] = P
    results["R"] = R
    results["F1"] = F1
    return results