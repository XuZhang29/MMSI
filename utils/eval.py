import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def classify_eval(results):
    predicted, true = results
    Acc = accuracy_score(true, predicted)
    P = precision_score(true, predicted)
    R = recall_score(true, predicted)
    F1 = f1_score(true, predicted)
    
    return Acc, P, R, F1

def regression_eval(results):
    predicted, true = results
    
    def ImpScore(ps, gs):
    
        def Score_map(difference):
            conditions = [
                (difference <= 0.2),
                (difference <= 0.4),
                (difference <= 0.6),
                (difference <= 0.8),
                (difference <= 1.0)
            ]
            values = [1.0, 0.8, 0.6, 0.4, 0.2]
            return np.select(conditions, values, default=0.0)
        scores = []
        for p, g in zip(ps, gs):
            scores.append(Score_map(np.abs(np.log(p + 1) - np.log(g + 1))))
            
        return np.mean(scores)

    def ImpAcc(ps, gs):
        accs = []
        for p, g in zip(ps, gs):
            if abs(p - g) / g <= 0.25:
                accs.append(1)
            else:
                accs.append(0)
                
        return np.mean(accs)
        

    def ImpErr(ps, gs):
        errs = []
        for p, g in zip(ps, gs):
            errs.append(abs(p - g) / 180)
            
        return np.mean(errs)

    return ImpScore(predicted, true), ImpAcc(predicted, true), ImpErr(predicted, true)

