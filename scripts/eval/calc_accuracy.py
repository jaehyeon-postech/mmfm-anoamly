from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np  
from pathlib import Path


def cal_metrics_miic(ground_truths, responses, penalize_missing=False):
    """
    penalize_missing=True  -> 미응답(-1)을 항상 오답으로 처리 (현재 방식)
    penalize_missing=False -> 응답 있는 샘플만으로 메트릭 계산하고,
                              커버리지(응답 비율)를 함께 출력
    """
    n_gt, n_resp = len(ground_truths), len(responses)
    print(n_resp, n_gt)

    # 길이 맞추기
    if n_resp < n_gt:
        responses = responses + ["__MISSING__"] * (n_gt - n_resp)
    elif n_resp > n_gt:
        responses = responses[:n_gt]

    y_true = [1 if gt else 0 for gt in ground_truths]
    y_pred = []
    for r in responses:
        r = (r or "").strip()
        r = r.split(",", 1)[0].strip()
        rl = r.lower()
    
        if r == "__MISSING__":
            y_pred.append(-1)
        elif "yes" in rl:
            y_pred.append(1)
        elif "no" in rl:
            y_pred.append(0)
        else:
            y_pred.append(-1)

    if penalize_missing:
        # Accuracy: -1은 무조건 오답
        correct = sum(1 for t, p in zip(y_true, y_pred) if p in (0, 1) and p == t)
        acc = correct / len(y_true) if y_true else 0.0

        # 정밀/재현/F1: -1을 항상 불리하게 매핑
        y_pred_for_metrics = []
        for t, p in zip(y_true, y_pred):
            if p == -1:
                y_pred_for_metrics.append(0 if t == 1 else 1)
            else:
                y_pred_for_metrics.append(p)

        precision = precision_score(y_true, y_pred_for_metrics, zero_division=0) if y_true else 0.0
        recall    = recall_score(y_true, y_pred_for_metrics, zero_division=0)    if y_true else 0.0
        f1        = f1_score(y_true, y_pred_for_metrics, zero_division=0)        if y_true else 0.0

        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print(f"Totals -> N: {len(y_true)}, Pos: {sum(y_true)}, Neg: {len(y_true)-sum(y_true)}")
        return acc, precision, recall, f1

    else:
        # 응답 있는 샘플만 평가
        valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
        coverage = len(valid_idx) / len(y_true) if y_true else 0.0

        if not valid_idx:
            print("No valid responses. Coverage: 0.0")
            return 0.0, 0.0, 0.0, 0.0

        yt = [y_true[i] for i in valid_idx]
        yp = [y_pred[i] for i in valid_idx]

        acc = sum(1 for t, p in zip(yt, yp) if p == t) / len(yt)
        precision = precision_score(yt, yp, zero_division=0)
        recall    = recall_score(yt, yp, zero_division=0)
        f1        = f1_score(yt, yp, zero_division=0)

        print("Coverage:", coverage)  # 응답이 있는 비율
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print(f"Evaluated on {len(yt)} / {len(y_true)} samples")
        return acc, precision, recall, f1

def cal_metrics(gt_data, responses):

    class_name = set()
    for item in gt_data:
        class_name.add(item['image'].split('/')[1])

    class_name = list(class_name)

    acc_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    num_anomaly = 0
    num_normal = 0
    for cls in class_name:
        y_true = []
        y_pred = []
        for index, item in enumerate(gt_data):
            if item['image'].split('/')[1] == cls:
                anomaly = 1 if item['anomaly'] else 0
                y_true.append(anomaly)
                pred = 1 if 'Yes' in responses[index] or 'yes' in responses[index] else 0
                y_pred.append(pred)
        
        acc = accuracy_score(y_true, y_pred)
        acc_results.append(acc)
        precision = precision_score(y_true, y_pred)
        precision_results.append(precision)
        recall = recall_score(y_true, y_pred)
        recall_results.append(recall)
        f1 = f1_score(y_true, y_pred)
        f1_results.append(f1)
        print(f'Class: {cls}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}')

        print(len(y_true))

        num_anomaly += sum(y_true)
        num_normal += len(y_true) - sum(y_true)

    print(f'Mean Accuracy: {sum(acc_results) / len(acc_results)}')
    print(f'Mean Precision: {sum(precision_results) / len(precision_results)}')
    print(f'Mean Recall: {sum(recall_results) / len(recall_results)}')
    print(f'Mean F1: {sum(f1_results) / len(f1_results)}')

    print(len(class_name))
    print(num_anomaly)
    print(num_normal)
