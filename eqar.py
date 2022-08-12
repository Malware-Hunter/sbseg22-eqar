import sys
import os
import argparse
import numpy as np
import pandas as pd
import time
import timeit
import csv
import shutil
from termcolor import colored, cprint
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from fim import apriori, eclat, fpgrowth
from functools import partial
from spinner import Spinner
#-----------------------------------------------------------------------
#Dictionary For Column Names
report_colnames = {
    'a': 'support_itemset_absolute',
    's': 'support_itemset_relative',
    'S': 'support_itemset_relative_pct',
    'b': 'support_antecedent_absolute',
    'x': 'support_antecedent_relative',
    'X': 'support_antecedent_relative_pct',
    'h': 'support_consequent_absolute',
    'y': 'support_consequent_relative',
    'Y': 'support_consequent_relative_pct',
    'c': 'confidence',
    'C': 'confidence_pct',
    'l': 'lift',
    'L': 'lift_pct',
    'e': 'evaluation',
    'E': 'evaluation_pct',
    'Q': 'xx',
    'S': 'support_emptyset',
}

DIR_BASE = ""
DIR_QFY = ""
DIR_TH = ""

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='str',
        help='Dataset (csv file).', type=str, required=True)
    parser.add_argument(
        '--use-balanced-datasets', help = "Use Balanced Datasets.",
        action = 'store_true')
    parser.add_argument(
        '-a', '--algorithm', metavar='str',
        choices=['apriori', 'fpgrowth', 'eclat'],
        type=str, default='eclat')
    parser.add_argument(
        '-m', '--max-length', metavar='int',
        help='Max length of rules (antecedent + consequent) (default: 5).',
        type=int, default=5)
    parser.add_argument(
        '-s', '--min-support', metavar='float',
        help='Minimum support ratio (must be > 0, default: 0.1).',
        type=float, default=0.1)
    parser.add_argument(
        '-c', '--min-confidence', metavar='float',
        help='Minimum confidence (default: 0.5).',
        type=float, default=0.5)
    parser.add_argument(
        '-l', '--min-lift', metavar='float',
        help='Minimum lift (default: 0.0).',
        type=float, default=0.0)
    parser.add_argument(
        '-t', '--threshold', metavar='float',
        help='''Without -q/--qualify: Minimum rules threshold to consider sample as malware (default: 0.0).
With -q/--qualify: Percentage of rules to be considered after qualification (default: 0.2).''',
        type=float, default=0.0)
    parser.add_argument(
        '-v', '--verbose', help="Increase Output Data.",
        action="store_true")
    parser.add_argument(
        '-q', '--qualify', metavar='str',
        help="Metric For Rules Qualification.",
        choices=['acc', 'c1', 'c2', 'bc', 'kap', 'zha', 'wl', 'corr', 'cov', 'prec'],
        type=str, default='')
    parser.add_argument(
        '-o', '--overwrite', help="Delete All Previous Data.",
        action="store_true")
    parser.add_argument(
        '-r', '--reduce', nargs='+', help="Reduce Dataset Data. Amount (Malwares Bemigns) Samples.",
        type=int)
    args = parser.parse_args(argv)
    return args

def to_pandas_dataframe(data, report):
    colnames = ['consequent', 'antecedent'] + [report_colnames.get(r, r) for r in list(report)]
    df = pd.DataFrame(data, columns=colnames)
    df = df.sort_values('support_itemset_relative', ascending=False)
    return df

def generate_unique_rules(dataset_df, lift, dataset_type):
    rules = []
    if not dataset_df.empty:
        r = dataset_df[(dataset_df['lift'] >= lift)]
        print("Deleting", dataset_type, "Repeated Rules")
        s = r[['consequent', 'antecedent']].values.tolist()
        r_list = [sorted({i[0]} | set(i[1])) for i in s]
        rules = list(map(list, set(map(lambda i: tuple(i), r_list))))
    return rules

def parallelize_func(func, parameters, const_parameter = None, cores = cpu_count()):
    parameters_split = np.array_split(parameters, cores)
    pool = Pool(cores)
    result = []
    if const_parameter == None:
        result.append(pool.map(func, parameters_split))
    else:
        result.append(pool.map(partial(func, const_parameter = const_parameter), parameters_split))
    pool.close()
    pool.join()
    return result[0]

def dataset_transaction(dataset):
    num_rows = dataset.shape[0]
    num_cols = dataset.shape[1]
    t = []
    for i in tqdm(range(0,num_rows)):
        l = []
        for j in range(0,num_cols):
            if dataset.values[i][j] != 0:
                l.append(j)
        t.append(l)
    return t

def to_fim_format(dataset):
    result = parallelize_func(dataset_transaction, dataset)
    data = []
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            data.append(result[i][j])
    return data

def execute_fim(parameters):
    p = parameters[0,:]
    algorithm = p[1].algorithm
    support = p[1].min_support * 100
    confidence = p[1].min_confidence * 100
    report = 'scl'
    print("Running FIM: Generating", p[2], "Association Rules")
    if algorithm == "apriori":
        result = apriori(p[0], target='r', zmin=2, zmax=p[1].max_length, supp=support,
                            conf=confidence, report=report, mode='o')
    elif algorithm == "fpgrowth":
        result = fpgrowth(p[0], target='r', zmin=2, zmax=p[1].max_length, supp=support,
                            conf=confidence, report=report, mode='o')
    elif algorithm == "eclat":
        result = eclat(p[0], target='r', zmin=2, zmax=p[1].max_length, supp=support,
                            conf=confidence, report=report, mode='o')
    r = to_pandas_dataframe(result, report)
    r = generate_unique_rules(r, p[1].min_lift, p[2])

    pct = (1.0 - (len(r)/len(result))) * 100.0 if len(result) != 0 else 0.0
    print("Generate {}".format(len(r)), p[2], "Association Rules ({:.3f})".format(pct))
    return r

def rules_difference(list_a, list_b):
    a = set(map(tuple, list_a))
    b = set(map(tuple, list_b))
    list_d = a -b
    return list_d

#def rules_subset(list_a, list_b, max_l):
def rules_subset(list_a, const_parameter):
    list_b = const_parameter[0]
    max_l = const_parameter[1]
    l = []
    for i in range(2, max_l):
        print("Processing Rules of Length", i)
        la = [r for r in list_a if len(r) == i]
        lb = [r for r in list_b if len(r) > i]
        for rule in tqdm(la):
            li = [r for r in lb if rule[0] <= r[0] and rule[-1] >= r[-1]]
            is_subset = any(set(rule).issubset(j) for j in li)
            if not is_subset:
                l.append(rule)
    la = [r for r in list_a if len(r) == max_l]
    l += la
    return l

#def self_rules_superset(list_a, max_l):
def self_rules_superset(list_a, const_parameter):
    list_b = const_parameter[0]
    max_l = const_parameter[1]
    l = []
    for i in range(3, max_l + 1):
        print("Processing Rules of Length", i)
        la = [r for r in list_a if len(r) == i]
        lb = [r for r in list_b if len(r) < i]
        for rule in tqdm(la):
            li = [r for r in lb if rule[0] <= r[0] and rule[-1] >= r[-1]]
            is_superset = any(set(rule).issuperset(j) for j in li)
            if not is_superset:
                l.append(rule)
    la = [r for r in list_a if len(r) == 2]
    l += la
    return l

def test_apps(test, const_parameter):
    rules = const_parameter[0]
    args = const_parameter[1]
    #class_ = test['class']
    features_test = test.drop(['class'], axis=1)
    #threshold = 0.0 if args.qualify else args.threshold
    pct_math_rules = []
    prediction_list = []
    for i in tqdm(range(0, len(test))):
        count_rules = 0
        for r in rules:
            app_ft = features_test.values[i,:]
            if len(r) ==  sum(x > 0 for x in app_ft[list(r)]): #sum(app_ft[list(r)]):
                count_rules += 1

        p = count_rules/len(rules)
        pct_math_rules.append(p)
    return pct_math_rules

def r_accuracy(p, n, P, N):
    return (p + (N - n)) / (P + N)

def r_coverage(p, n, P, N):
    return (p + n) / (P + N)

def r_precision(p, n, P, N):
    return p / (p + n)

def r_logical_sufficiency(p, n, P, N):
    a = p * N
    b = n * P
    if b == 0:
        b = 0.0001
    return a / b

def r_bayesian_confirmation(p, n, P, N):
    a = p / (p + N)
    b = P - p
    c = P -p + N - n
    return a - (b / c)

def r_kappa(p, n, P, N):
    a = (p * N) - (P * n)
    b = (P + N) * (p + n + P)
    c = 2 * (p + n) * P
    return 2 * (a / (b - c))

def r_zhang(p, n, P, N):
    a = (p * N) - (P * n)
    b = p * N
    c = P * n
    return a / max([b, c])

def r_correlation(p, n, P, N):
    import math
    a = (p * N) - (P * n)
    b = P * N * (p + n)
    c = P -p + N - n
    r = math.sqrt(b * c)
    return a / r

def r_wlaplace(p, n, P, N):
    a = (p + 1) * (P + N)
    b = (p + n + 2) * P
    return a / b

def coleman(p, n, P, N):
    a = ((P + N) * (p / (p + n))) - P
    return a / N

def cohen(p, n, P, N):
    a = ((P + N) * (p / (p + n))) - P
    b = (P + N) / 2
    c = (p + n + P) / (p + n)
    return a / ((b * c) - P)

def r_c1(p, n, P, N):
    a = coleman(p, n, P, N)
    b = cohen(p, n, P, N)
    return a * ((2 + b) / 3)

def r_c2(p, n, P, N):
    a = coleman(p, n, P, N)
    b = 1 + (p / P)
    return a * 0.5 * b

def quality_par(rules, const_parameter):
    dataset = const_parameter[0]
    rules_metrics = []
    features = dataset.drop(['class'], axis=1)
    for r in tqdm(rules):
        rule_coverage = 0 # == p + n
        p = 0
        for i in range(0, len(dataset)):
            app_ft = features.values[i,:]
            if len(r) ==  sum(x > 0 for x in app_ft[list(r)]):
                rule_coverage += 1
                class_ = dataset.values[i,:]
                class_ = class_[-1]
                if class_ == 1:
                    p += 1
        n = rule_coverage - p
        rules_metrics.append([p, n])
    return rules_metrics

def file_content(dir_name, fold_no, file):
    f_name = dir_name + "/" + str(fold_no) + "_" + file
    ct = []
    try:
        with open(f_name) as f:
            if file == "log":
                ct = [l for l in f]
                ct = ct[0]
            elif file in ["mw_rules", "bw_rules", "qualify_parameters"]:
                ct = [list(map(int, l.split(','))) for l in f]
            elif file in ["rules_diff", "rules_subset", "rules_superset", "rules_qualify"]:# or file.startswith("rules_qualify"):
                ct = [tuple(map(int, l.split(','))) for l in f]
            elif file == "test_apps":
                ct = [list(map(float, l.split(','))) for l in f]
                ct = ct[0]
            elif file == "times":
                ct = [(l.split(',')[0], float(l.split(',')[1])) for l in f]
        return ct
    except BaseException as e:
        #print(e)
        return ct

def update_log(dir_name, fold_no, step):
    f_name = dir_name + "/" + str(fold_no) + "_log"
    with open(f_name, 'w') as f:
        f.write(step)

def save_to_file(obj_list, dir_name, fold_no, step):
    f_name = dir_name + "/" + str(fold_no) + "_" + step
    with open(f_name,"w", newline='') as f:
        f_writer = csv.writer(f)
        for obj in obj_list:
            f_writer.writerow(obj)

step_dict = {
    "rules_generate": 0,
    "rules_diff": 1,
    "rules_subset": 2,
    "rules_superset": 3,
    "rules_qualify": 4,
    "test_apps": 5,
    "results_calc": 6,
    "finished": 7
}

def get_rules(train, args, fold_no):
    step = file_content(DIR_BASE, fold_no, "log")
    step = step if len(step) else "rules_generate"
    stopped_step = step_dict.get(step)
    time_l = []

    step = "finished"
    if stopped_step == step_dict.get(step):
        update_log(DIR_BASE, fold_no, step)
        rules = file_content(DIR_BASE, fold_no, "rules_superset")
        time_l = file_content(DIR_BASE, fold_no, "times")
        return rules, time_l

    step = "rules_generate"
    mw_rules = []
    bw_rules = []
    if stopped_step <= step_dict.get(step):
        update_log(DIR_BASE, fold_no, step)
        mw_dataset = train[(train['class'] == 1)]
        mw_dataset = mw_dataset.drop(['class'], axis=1)
        bw_dataset = train[(train['class'] == 0)]
        bw_dataset = bw_dataset.drop(['class'], axis=1)

        spn = Spinner('Preparing MALWARES Data')
        spn.start()
        mw_fim = to_fim_format(mw_dataset)
        spn.stop()

        spn = Spinner('Preparing BENIGNS Data')
        spn.start()
        bw_fim = to_fim_format(bw_dataset)
        spn.stop()

        p = np.array([[mw_fim, args, "MALWARES"], [bw_fim, args, "BENIGNS"]], dtype=object)

        start = timeit.default_timer()
        rules = parallelize_func(execute_fim, p, cores = 2)
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        save_to_file(time_l, DIR_BASE, fold_no, "times")
        mw_rules = rules[0]
        save_to_file(mw_rules, DIR_BASE, fold_no, "mw_rules")
        bw_rules = rules[1]
        save_to_file(bw_rules, DIR_BASE, fold_no, "bw_rules")

    step = "rules_diff"
    rules = []
    if stopped_step == step_dict.get(step):
        mw_rules = file_content(DIR_BASE, fold_no, "mw_rules")
        bw_rules = file_content(DIR_BASE, fold_no, "bw_rules")
        time_l = file_content(DIR_BASE, fold_no, "times")
        time_l = time_l[:stopped_step]

    if stopped_step <= step_dict.get(step):
        update_log(DIR_BASE, fold_no, step)
        print("Generating Unique MALWARES Rules: Difference")
        start = timeit.default_timer()
        rules = rules_difference(mw_rules, bw_rules)
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        save_to_file(time_l, DIR_BASE, fold_no, "times")
        save_to_file(rules, DIR_BASE, fold_no, step)

    step = "rules_subset"
    if stopped_step == step_dict.get(step):
        mw_rules = file_content(DIR_BASE, fold_no, "mw_rules")
        bw_rules = file_content(DIR_BASE, fold_no, "bw_rules")
        rules = file_content(DIR_BASE, fold_no, "rules_diff")
        time_l = file_content(DIR_BASE, fold_no, "times")
        time_l = time_l[:stopped_step]

    if stopped_step <= step_dict.get(step):
        update_log(DIR_BASE, fold_no, step)
        print("Generating Unique MALWARES Rules: Subset")
        start = timeit.default_timer()
        rules = list(rules)
        p = np.array(rules, dtype=object)
        result = parallelize_func(rules_subset, p, const_parameter = [bw_rules, args.max_length])
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        rules = [l for r in result for l in r]
        save_to_file(time_l, DIR_BASE, fold_no, "times")
        save_to_file(rules, DIR_BASE, fold_no, step)

    step = "rules_superset"
    if stopped_step == step_dict.get(step):
        rules = file_content(DIR_BASE, fold_no, "rules_subset")
        time_l = file_content(DIR_BASE, fold_no, "times")
        time_l = time_l[:stopped_step]

    if stopped_step <= step_dict.get(step):
        update_log(DIR_BASE, fold_no, step)
        print("Generating Unique MALWARES Rules: Superset")
        start = timeit.default_timer()
        rules = list(rules)
        p = np.array(rules, dtype=object)
        result = parallelize_func(self_rules_superset, p, const_parameter = [rules, args.max_length])
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        rules = [l for r in result for l in r]
        save_to_file(time_l, DIR_BASE, fold_no, "times")
        save_to_file(rules, DIR_BASE, fold_no, step)

    step = "finished"
    update_log(DIR_BASE, fold_no, step)

    return rules, time_l

def get_results(test, rules, times, args, fold_no):
    step = file_content(DIR_TH, fold_no, "log")
    step = step if len(step) else "test_apps"
    stopped_step = step_dict.get(step)
    time_l = times

    step = "test_apps"
    pct_match_rules = []
    if stopped_step <= step_dict.get(step):
        if not rules:
            print(colored("There Are No Rules For Testing",'red'))
            return None, None
        print(colored("{} Rules Generated".format(len(rules)), 'green'))

        update_log(DIR_TH, fold_no, step)
        print("Testing Applications")
        start = timeit.default_timer()
        result = parallelize_func(test_apps, test, const_parameter = [rules, args])
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        save_to_file(time_l, DIR_TH, fold_no, "times")
        pct_match_rules  = [l for r in result for l in r]
        save_to_file([pct_match_rules], DIR_TH, fold_no, step)

    step = "results_calc"
    result_dict ={}
    test_prediction = []
    if stopped_step == step_dict.get(step):
        pct_match_rules = file_content(DIR_TH, fold_no, "test_apps")
        time_l = file_content(DIR_TH, fold_no, "times")

    threshold = 0.0 if args.qualify else args.threshold
    class_ = test['class']
    i = 0
    for p in pct_match_rules:
        prediction = 1 if p > threshold else 0
        if args.verbose:
            color = 'green' if class_.values[i] == prediction else 'red'
            ds_class = "MALWARE" if class_.values[i] else "BENIGN"
            model_class = "MALWARE" if prediction else "BENIGN"
            text = "APK " + str(i) + " >> Class: " + ds_class + " | Prediction: " + model_class
            text += " (" + "{:.3f}".format(p) + ")" #if prediction else ""
            print(colored(text, color))
            i += 1
        test_prediction.append(prediction)

    if stopped_step <= step_dict.get(step):
        update_log(DIR_TH, fold_no, step)
        test_classification = list(test['class'])
        result_dict = general_result(test_classification, test_prediction, len(rules))

    time_dict = dict(time_l)
    return result_dict, time_dict, test_prediction

#def quality_rules(rules, dataset, q_measure):
def quality_rules(rules, const_parameter):
    #Dictionary For Rules Qualify Measures
    rules_measures = {
        'acc': r_accuracy,
        'cov': r_coverage,
        'prec': r_precision,
        'ls': r_logical_sufficiency,
        'bc': r_bayesian_confirmation,
        'kap': r_kappa,
        "zha": r_zhang,
        'corr': r_correlation,
        'c1': r_c1,
        'c2': r_c2,
        'wl': r_wlaplace,
    }
    dataset = const_parameter[0]
    q_measure = const_parameter[1]
    P = len(dataset[(dataset['class'] == 1)])
    N = len(dataset[(dataset['class'] == 0)])
    rules_metrics = []
    func = rules_measures.get(q_measure, lambda: "Invalid Qualify Measure")
    for index, row in rules.iterrows():
        r = row['rule']
        p = row['qualify_parameters'][0]
        n = row['qualify_parameters'][1]
        #execute the function
        q = func(p, n, P, N)
        #print(q)
        rule_dict = {
            "rule": r,
            "q_value": q
        }
        rules_metrics.append(rule_dict)
    rdf = pd.DataFrame(rules_metrics)
    return rdf

def get_qualified_rules(train, rules, times, args, fold):
    step = file_content(DIR_QFY, fold_no, "log")
    step = step if len(step) else "rules_qualify"
    stopped_step = step_dict.get(step)
    time_l = times
    rules_l = []

    step = "finished"
    if stopped_step == step_dict.get(step):
        update_log(DIR_QFY, fold_no, step)
        rules_l = file_content(DIR_QFY, fold_no, "rules_qualify")
        time_l = file_content(DIR_QFY, fold_no, "times")
        return rules_l, time_l

    step = "rules_qualify"
    if stopped_step <= step_dict.get(step):
        update_log(DIR_QFY, fold_no, step)
        print("Qualifying MALWARES Rules")
        result = file_content(DIR_QFY, fold_no, "qualify_parameters")
        start = timeit.default_timer()
        if not len(result):
            p = np.array(rules, dtype=object)
            results = parallelize_func(quality_par, p, const_parameter = [train])
            result  = [l for r in results for l in r]
            save_to_file(result, DIR_QFY, fold_no, "qualify_parameters")
        df = pd.DataFrame(list(zip(rules, result)), columns =['rule', 'qualify_parameters'])
        results = parallelize_func(quality_rules, df, const_parameter = [train, args.qualify])
        rdf = pd.concat(results)
        rdf = rdf.sort_values(by=['q_value'], ascending=False)
        rules_l = list(rdf['rule'])
        rules_q = list(rdf['q_value'])
        end = timeit.default_timer()
        time_tuple = (step, end - start)
        time_l.append(time_tuple)
        save_to_file(time_l, DIR_QFY, fold_no, "times")
        save_to_file(rules_l, DIR_QFY, fold_no, step)
        print(rules_q[:15])

    step = "finished"
    update_log(DIR_QFY, fold_no, step)

    return rules_l, time_l

def eqar(train, test, args, fold_no):
    rules, time_l = get_rules(train, args, fold_no)

    if args.qualify:
        rules, time_l = get_qualified_rules(train, rules, time_l, args, fold_no)
        threshold = args.threshold if args.threshold else 0.2
        num_rules = int(len(rules) * threshold)
        rules = rules[:num_rules]

    evaluation_metrics, runtime, prediction = get_results(test, rules, time_l, args, fold_no)

    return evaluation_metrics, runtime, prediction

def format_msg(result):
    if not result:
        return "No Results."
    msg = "TP: {} TN: {} FP: {} FN: {}".format(result.get("tp"),result.get("tn"), result.get("fp"), result.get("fn"))
    precision = result.get("precision") * 100.0
    accuracy = result.get("accuracy") * 100.0
    recall = result.get("recall") * 100.0
    f1_score = result.get("f1_score") * 100.0
    mcc = result.get("mcc")
    roc_auc = result.get("roc_auc")
    msg += "\nAccuracy: {:.3f}".format(accuracy)
    msg += "\nPrecision: {:.3f}".format(precision) #Ability to Correctly Detect Malware
    msg += "\nRecall: {:.3f}".format(recall)
    msg += "\nF1 Score: {:.3f}".format(f1_score)
    msg += "\nMCC: {:.3f}".format(mcc)
    msg += "\nROC AuC: {:.3f}\n".format(roc_auc)
    print(colored(msg, 'yellow'))
    return msg

def balanced_dataset(dataset):
    B = dataset[(dataset['class'] == 0)]
    M = dataset[(dataset['class'] == 1)]

    lenB = len(B)
    lenM = len(M)
    b_dataset = None
    if lenB > lenM:
        random_select = B.sample(n = lenM, random_state = 0)
        b_dataset = pd.concat([random_select, M], ignore_index = True)
    else:
        random_select = M.sample(n = lenB, random_state = 0)
        b_dataset = pd.concat([B, random_select], ignore_index = True)
    #b_dataset.to_csv("balanced_" + args.dataset, index = False)
    return b_dataset

def general_result(test_classification, test_prediction, num_rules = -1):
    tn, fp, fn, tp = confusion_matrix(test_classification, test_prediction).ravel()
    accuracy = metrics.accuracy_score(test_classification, test_prediction)
    precision = metrics.precision_score(test_classification, test_prediction, zero_division = 0)
    recall = metrics.recall_score(test_classification, test_prediction, zero_division = 0)
    f1_score = metrics.f1_score(test_classification, test_prediction, zero_division = 0)
    mcc = metrics.matthews_corrcoef(test_classification, test_prediction)
    roc_auc = metrics.roc_auc_score(test_classification, test_prediction)

    result_dict = {
        "num_rules": num_rules,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mcc": mcc,
        "roc_auc": roc_auc
    }

    return result_dict

if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    try:
        dataset = pd.read_csv(args.dataset)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)

    # - ->> directories to save data <<- -#
    root_path = os.getcwd()
    dir_name = (args.dataset).split('.')[0]
    dir_name += "_S" + str(int(args.min_support * 100.0))
    dir_name += "_C" + str(int(args.min_confidence * 100.0))

    path = os.path.join(root_path, dir_name)
    DIR_BASE = path
    if args.overwrite and os.path.exists(path):
        shutil.rmtree(path)

    if args.qualify:
        path = os.path.join(path, args.qualify)
        DIR_QFY = path

    th_int = int(args.threshold * 100.0)
    th_str = "00" if th_int < 1 else str(th_int)
    th_dir = "T" + th_str

    DIR_TH = os.path.join(path, th_dir)
    if not os.path.exists(DIR_TH):
        os.makedirs(DIR_TH)
    # ---> <---#

    if args.reduce:
        m = dataset[(dataset['class'] == 1)]
        m = m.head(args.reduce[0])
        b = dataset[(dataset['class'] == 0)]
        b = b.head(args.reduce[1])
        frames = [m, b]
        dataset = pd.concat(frames, ignore_index=True)

    if args.use_balanced_datasets:
        print('Using Balanced Dataset')
        dataset = balanced_dataset(dataset)

    class_ = dataset['class']

    skf = StratifiedKFold(n_splits=5)
    fold_no = 1
    output_str = ""
    results_list = []
    times_list = []
    general_class = []
    general_pred = []
    for train_index, test_index in skf.split(dataset, class_):
        train = dataset.loc[train_index,:]
        test = dataset.loc[test_index,:]
        p = sum(test['class'])/len(test['class'])
        output_str += "\nFold {} - Class Ratio: {:.3f} ({})\n".format(fold_no, p, sum(train['class']))
        print(colored("\nFold {}".format(fold_no), 'green'))
        result, time_result, pred_result = eqar(train, test, args, fold_no)
        output_str += format_msg(result)
        results_list.append(result)
        times_list.append(time_result)
        general_class += list(test['class'])
        general_pred += pred_result
        fold_no += 1
    print(colored("\n>>>>> + <<<<<", 'green'))
    print(colored(output_str, 'yellow'))

    gr_dict = general_result(general_class, general_pred)
    print(colored("GENERAL RESULT", 'yellow'))
    gr_str = format_msg(gr_dict)

    text_file = open(DIR_TH + "/general_results.txt", "w")
    text_file.write(gr_str)
    text_file.close()

    rdf = pd.DataFrame(results_list)
    rdf.to_csv(DIR_TH + "/model_results.csv", index = False)
    tdf = pd.DataFrame(times_list)
    tdf.to_csv(DIR_TH + "/time_results.csv", index = False)
