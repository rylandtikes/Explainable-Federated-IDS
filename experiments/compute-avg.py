import re, sys, statistics as stats

metrics = {"acc":[], "prec":[], "rec":[], "auc":[]}

with open(sys.argv[1]) as f:
    for line in f:
        m = re.search(r'Accuracy: ([0-9.]+).*Precision: ([0-9.]+).*Recall: ([0-9.]+).*AUC: ([0-9.]+)', line)
        if m:
            metrics["acc"].append(float(m.group(1))*100)
            metrics["prec"].append(float(m.group(2))*100)
            metrics["rec"].append(float(m.group(3))*100)
            metrics["auc"].append(float(m.group(4))*100)

# compute averages
acc = stats.mean(metrics["acc"])
prec = stats.mean(metrics["prec"])
rec = stats.mean(metrics["rec"])
auc = stats.mean(metrics["auc"])
f1 = 2*prec*rec/(prec+rec)

print(f"Acc={acc:.2f}% Prec={prec:.2f}% Rec={rec:.2f}% F1={f1:.2f}% AUC={auc:.2f}%")

