from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np  


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
