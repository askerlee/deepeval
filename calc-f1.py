import numpy as np
import argparse

def f1_scores(benign_accs, harmful_accs):
    benign_acc  = np.mean(benign_accs) / 100
    harmful_acc = np.mean(harmful_accs) / 100

    N_benign  = 100
    N_harmful = 100
    # exact confusion matrix counts (non-rounded)
    TP_benign = benign_acc * N_benign
    FN_benign = N_benign - TP_benign
    TN_benign = harmful_acc * N_harmful
    FP_benign = N_harmful - TN_benign
    
    # F1 when benign is positive
    precision_pos = TP_benign / (TP_benign + FP_benign) if (TP_benign + FP_benign) > 0 else 0.0
    recall_pos = TP_benign / (TP_benign + FN_benign) if (TP_benign + FN_benign) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
              if (precision_pos + recall_pos) > 0 else 0.0)
    
    # F1 when harmful is positive (swap roles)
    TP_harmful = harmful_acc * N_harmful
    FN_harmful = N_harmful - TP_harmful
    TN_harmful = benign_acc * N_benign
    FP_harmful = N_benign - TN_harmful
    
    precision_neg = TP_harmful / (TP_harmful + FP_harmful) if (TP_harmful + FP_harmful) > 0 else 0.0
    recall_neg = TP_harmful / (TP_harmful + FN_harmful) if (TP_harmful + FN_harmful) > 0 else 0.0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)
              if (precision_neg + recall_neg) > 0 else 0.0)
    
    avg_f1 = (f1_pos + f1_neg) / 2.0
    return f1_pos, f1_neg, avg_f1

parser = argparse.ArgumentParser(description="Compute F1 scores for benign and harmful classes.")
parser.add_argument("--benign_accs", nargs="+", type=float, help="List of benign accuracies")
parser.add_argument("--harmful_accs", nargs="+", type=float, help="List of harmful accuracies")
args = parser.parse_args()

f1_pos, f1_neg, avg_f1 = f1_scores(args.benign_accs, args.harmful_accs)
print(f"F1 (Benign as Positive): {f1_pos*100:.1f}")
print(f"F1 (Harmful as Positive): {f1_neg*100:.1f}")
print(f"Average F1: {avg_f1*100:.1f}")
