"""Shared code for the split SelfCheckGPT replication notebooks.

Holds everything that is cheap and model-free: configuration, dataset, human labels,
evaluation metrics, paper reference numbers, cache helpers. No model is loaded here.
"""
import os, gc, time, pickle, functools, json, warnings
from collections import Counter
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.environ.get("SC_RESULTS_DIR", "./selfcheckgpt_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# CONFIGURATION (shared by all notebooks (01-06))
# ----------------------------------------------------------------------------------
SEED = 42

RUN_NGRAM_ORDERS     = True   # Table 7 : n = 1..5, Avg / Max
RUN_WIKIBIO_EVIDENCE = True   # Table 3 : WikiBio paragraph as evidence instead of the 20 samples
RUN_SAMPLE_ABLATION  = True   # Fig 7/8 : performance vs number of samples
VERIFY_AGAINST_PACKAGE = True # re-implemented scorers are checked against the official package on a few sentences

MAX_PASSAGES        = None    # all 238 passages 
MQAG_MAX_PASSAGES   = 20
PROMPT_MAX_PASSAGES = 30

MQAG_NUM_QUESTIONS = 5
MQAG_FP16 = True

LLM_BACKEND    = "hf"                                  # "hf" = open-source LLM, "openai" = OpenAI API
HF_LLM         = "mistralai/Mistral-7B-Instruct-v0.2"
HF_4BIT        = "auto"                                # "auto": 4-bit NF4 if GPU memory < 20 GB; or True / False
PROMPT_BATCH   = 20                                   
OPENAI_MODEL   = "gpt-3.5-turbo"
OPENAI_WORKERS = 16

NS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]           # sample counts for the Fig 7/8 ablation

np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------------------------------------------------------------
# device / memory helpers
# ----------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == "cuda" else 0.0

def describe_device():
    print(f"Using device: {device}")
    if device.type == "cuda": print(f"{torch.cuda.get_device_name(0)} - {GPU_GB:.1f} GB")
    else: print("WARNING: no GPU found - Runtime > Change runtime type > GPU.")
    print("Results / caches ->", RESULTS_DIR)

def ram(tag=""):
    # print this process RAM and the machine's free RAM (use it after loading a model to see how close to the limit)
    import psutil
    vm = psutil.virtual_memory(); me = psutil.Process().memory_info().rss / 2**30
    print(f"[RAM {tag}] this process {me:.1f} GB | system used {vm.used / 2**30:.1f} / {vm.total / 2**30:.1f} GB")

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def get_nlp():
    # spaCy model; selfcheckgpt re-loads it on every call, so cache spacy.load 
    import spacy
    if not hasattr(spacy.load, "cache_info"):
        spacy.load = functools.lru_cache(maxsize=None)(spacy.load)
    return spacy.load("en_core_web_sm")

# ----------------------------------------------------------------------------------
# resumable per-passage caching (files live in RESULTS_DIR on Google Drive)
# ----------------------------------------------------------------------------------
def cache_path(name): return os.path.join(RESULTS_DIR, f"cache_{name}.pkl")

def load_cache(name):
    p = cache_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f: return pickle.load(f)
    return {}

def save_cache(name, obj):
    p = cache_path(name); tmp = p + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, p)

def run_cached(name, fn, ids, desc, save_every=10):
    # compute fn(passage_id) for every id, skipping ids already stored in the cache file
    cache = load_cache(name)
    todo = [i for i in ids if i not in cache]
    print(f"{desc}: {len(ids) - len(todo)} cached, {len(todo)} to compute")
    for k, i in enumerate(tqdm(todo, desc=desc)):
        cache[i] = fn(i)
        if (k + 1) % save_every == 0: save_cache(name, cache)
    if todo: save_cache(name, cache)
    return {i: cache[i] for i in ids}

def load_complete(name, ids):
    # used by the results notebook: return the cached dict only if it covers every requested passage
    c = load_cache(name)
    missing = [i for i in ids if i not in c]
    if missing:
        print(f"  [skipped] cache '{name}' is missing {len(missing)}/{len(ids)} passages -> run the notebook that produces it first")
        return None
    return {i: c[i] for i in ids}

def check_close(label, a, b, atol):
    ok = np.allclose(a, b, atol=atol)
    print(f"[{'OK' if ok else 'MISMATCH'}] {label}: max |diff| = {np.max(np.abs(np.asarray(a) - np.asarray(b))):.2e} (tol {atol})")
    return ok

# ----------------------------------------------------------------------------------
# dataset + human labels
# ----------------------------------------------------------------------------------
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
data = dataset["evaluation"]
rows = [data[i] for i in range(len(data))]
ALL_IDS = list(range(len(rows)))
IDS = ALL_IDS[:MAX_PASSAGES]
MQAG_IDS = IDS[:MQAG_MAX_PASSAGES]
PROMPT_IDS = IDS[:PROMPT_MAX_PASSAGES]
PROMPT_NAME = HF_LLM.split("/")[-1] if LLM_BACKEND == "hf" else OPENAI_MODEL
PROMPT_SLUG = PROMPT_NAME.replace("/", "_")

LABEL_VALUE = {"accurate": 0.0, "minor_inaccurate": 0.5, "major_inaccurate": 1.0}
HUMAN = {i: np.array([LABEL_VALUE[a] for a in r["annotation"]]) for i, r in enumerate(rows)}
for i, r in enumerate(rows):
    assert len(r["gpt3_sentences"]) == len(HUMAN[i]), f"sentence/label mismatch in passage {i}"
    assert len(r["gpt3_text_samples"]) == 20, f"passage {i} does not have 20 samples"

def get_samples(i, evidence="samples"):
    # evidence = "samples": the 20 stochastic GPT-3 samples; "wiki": the WikiBio reference paragraph (Table 3 ablation)
    r = rows[i]
    return list(r["gpt3_text_samples"]) if evidence == "samples" else [r["wiki_bio_text"]]

METRICS = ["NonFact", "NonFact*", "Factual", "Pearson", "Spearman"]

def auc_pr(y_true, y_score):
    # AUC-PR: trapezoidal area under precision_recall_curve
    P, R, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    return auc(R, P)

def _cat(sent_scores, ids):
    if len(ids) == 0: return np.zeros(0), np.zeros(0)
    s = np.concatenate([np.asarray(sent_scores[i], dtype=float) for i in ids])
    y = np.concatenate([HUMAN[i] for i in ids])
    assert len(s) == len(y), "score / label length mismatch"
    if not np.isfinite(s).all(): raise ValueError("non-finite sentence scores found")
    return s, y

def task_arrays(sent_scores, ids=None):
    # {task: (binary labels, detection score)} for the three sentence-level tasks
    ids = sorted(sent_scores) if ids is None else list(ids)
    s, y = _cat(sent_scores, ids)
    s2, y2 = _cat(sent_scores, [i for i in ids if HUMAN[i].mean() < 0.99])
    return {"NonFact":  ((y > 0.499).astype(int),  s),
            "NonFact*": ((y2 > 0.99).astype(int),  s2),
            "Factual":  ((y < 0.499).astype(int), -s)}

def passage_arrays(sent_scores, passage_scores=None, ids=None):
    ids = sorted(sent_scores) if ids is None else list(ids)
    h = np.array([HUMAN[i].mean() for i in ids])
    p = (np.array([np.mean(sent_scores[i]) for i in ids]) if passage_scores is None
         else np.array([passage_scores[i] for i in ids]))
    return p, h

def evaluate(sent_scores, passage_scores=None, ids=None):
    ids = sorted(sent_scores) if ids is None else list(ids)
    res = {}
    for task, (y, s) in task_arrays(sent_scores, ids).items():
        res[task] = 100 * auc_pr(y, s) if (len(y) and y.sum() > 0) else np.nan
    p, h = passage_arrays(sent_scores, passage_scores, ids)
    res["Pearson"]  = 100 * pearsonr(p, h)[0]  if len(ids) > 2 else np.nan
    res["Spearman"] = 100 * spearmanr(p, h)[0] if len(ids) > 2 else np.nan
    res["n_passages"] = len(ids)
    return res

def random_baseline(ids):
    ids = list(ids); y = {k: v[0] for k, v in task_arrays({i: np.zeros(len(HUMAN[i])) for i in ids}, ids).items()}
    return {k: 100 * float(np.mean(v)) if len(v) else np.nan for k, v in y.items()}

def show_df(df, fmt="{:.2f}"):
    try:
        from IPython.display import display as _d
        _d(df.style.format(fmt, subset=[c for c in METRICS if c in df.columns], na_rep="-").hide(axis="index"))
    except Exception:
        print(df.round(2).to_string(index=False))

def show_comparison(res, ref=None, title=None):
    # table of replicated metrics, with the paper's number and the difference underneath when available
    ref = ref or {}
    recs = []
    for name, m in res.items():
        recs.append({"Method": name, "Source": "this notebook", **{k: m[k] for k in METRICS}, "n_passages": m.get("n_passages")})
        if ref.get(name) is not None:
            recs.append({"Method": "", "Source": ref[name][0], **dict(zip(METRICS, ref[name][1])), "n_passages": None})
            recs.append({"Method": "", "Source": "difference (ours - ref)",
                         **{k: (m[k] - v if v is not None else np.nan) for k, v in zip(METRICS, ref[name][1])}, "n_passages": None})
    df = pd.DataFrame(recs)
    for k in METRICS: df[k] = pd.to_numeric(df[k], errors="coerce")
    if title: print(title)
    show_df(df)
    return df

# ---------reference numbers copied from the paper ----------
PAPER_T2 = {   # NonFact, NonFact*, Factual, Pearson, Spearman
    "SelfCheck-BERTScore":       ("paper Table 2", (81.96, 45.96, 44.23, 58.18, 55.90)),
    "SelfCheck-QA":              ("paper Table 2", (84.26, 40.06, 48.14, 61.07, 59.29)),
    "SelfCheck-Unigram (max)":   ("paper Table 2", (85.63, 41.04, 58.47, 64.71, 64.91)),
    "SelfCheck-NLI":             ("paper Table 2", (92.50, 45.17, 66.08, 74.14, 73.78)),
}
PAPER_PROMPT_CHATGPT = ("paper Table 2 (ChatGPT)", (93.42, 53.19, 67.09, 78.32, 78.30))
README_PROMPT = {   # official README, open-source LLMs (NonFact, Factual, PCC only)
    "Mistral-7B-Instruct-v0.2": ("README (Mistral-7B)",  (91.31, None, 62.76, 74.46, None)),
    "Llama-2-7b-chat-hf":       ("README (Llama2-7B)",   (89.05, None, 63.06, 61.52, None)),
    "Llama-2-13b-chat-hf":      ("README (Llama2-13B)",  (91.91, None, 64.34, 75.44, None)),
}
PAPER_T3 = {   # external knowledge ablation
    "SelfCheck-BERTScore":     ("paper Table 3", (81.96, 45.96, 44.23, 58.18, 55.90)),
    "WikiBio+BERTScore":       ("paper Table 3", (81.32, 40.62, 49.15, 58.71, 55.80)),
    "SelfCheck-QA":            ("paper Table 3", (84.26, 40.06, 48.14, 61.07, 59.29)),
    "WikiBio+QA":              ("paper Table 3", (84.18, 45.40, 52.03, 57.26, 53.62)),
    "SelfCheck-Unigram (max)": ("paper Table 3", (85.63, 41.04, 58.47, 64.71, 64.91)),
    "WikiBio+Unigram (max)":   ("paper Table 3", (80.43, 31.47, 40.53, 28.67, 26.70)),
    "SelfCheck-NLI":           ("paper Table 3", (92.50, 45.17, 66.08, 74.14, 73.78)),
    "WikiBio+NLI":             ("paper Table 3", (91.18, 48.14, 71.61, 78.84, 80.00)),
    "SelfCheck-Prompt (ChatGPT)": ("paper Table 3", (93.42, 53.19, 67.09, 78.32, 78.30)),
    "WikiBio+Prompt (ChatGPT)":   ("paper Table 3", (93.59, 65.26, 73.11, 85.90, 86.11)),
}
PAPER_T5 = {   # QA scoring variants
    "QA SimpleCount": ("paper Table 5", (83.97, 40.07, 47.78, 57.39, 55.15)),
    "QA +Bayes":      ("paper Table 5", (83.04, 38.58, 47.41, 56.43, 55.03)),
    "QA +Bayes+alpha":("paper Table 5", (84.26, 40.06, 48.14, 61.07, 59.29)),
}
PAPER_T7 = {   # n-gram orders
    "Avg(-logp) 1-gram": ("paper Table 7", (81.52, 40.33, 41.76, 40.68, 39.22)),
    "Avg(-logp) 2-gram": ("paper Table 7", (82.94, 44.38, 52.81, 58.84, 58.11)),
    "Avg(-logp) 3-gram": ("paper Table 7", (83.56, 44.64, 53.99, 62.21, 63.00)),
    "Avg(-logp) 4-gram": ("paper Table 7", (83.80, 43.55, 54.25, 61.98, 63.64)),
    "Avg(-logp) 5-gram": ("paper Table 7", (83.45, 42.31, 53.98, 60.68, 62.96)),
    "Max(-logp) 1-gram": ("paper Table 7", (85.63, 41.04, 58.47, 64.71, 64.91)),
    "Max(-logp) 2-gram": ("paper Table 7", (85.26, 39.29, 58.29, 62.48, 66.04)),
    "Max(-logp) 3-gram": ("paper Table 7", (84.97, 37.10, 57.08, 57.34, 60.49)),
    "Max(-logp) 4-gram": ("paper Table 7", (84.49, 36.37, 55.96, 55.77, 57.25)),
    "Max(-logp) 5-gram": ("paper Table 7", (84.12, 36.19, 54.89, 54.84, 55.97)),
}

def scores_from_matrices(variant, mats, n=None):
    # sentence score = mean over the first n samples (BERTScore: 1 - mean of best-match F1)
    out = {}
    for i, M in mats.items():
        Mn = M if n is None else M[:, :n]
        out[i] = (1.0 - Mn.mean(axis=1)) if variant == "bertscore" else Mn.mean(axis=1)
    return out


# ----------------------------------------------------------------------------------
# registries + helper used by the per-variant notebooks (each notebook prints its own quick result)
# ----------------------------------------------------------------------------------
RES_MAIN, SENT_MAIN, PASS_MAIN = {}, {}, {}    # Table 2: metrics / sentence scores / passage scores per method
RES_WIKI, RES_QA, RES_NGRAM = {}, {}, {}       # Table 3 / Table 5 / Table 7
MATS = {"bertscore": {}, "nli": {}, "prompt": {}}   # variant -> evidence -> {passage_id: sentences x samples matrix}
QCACHE = {}                                    # evidence -> {passage_id: per-sentence list of cached MQAG questions}

def register_main(name, sent_scores, passage_scores=None):
    RES_MAIN[name] = evaluate(sent_scores, passage_scores)
    SENT_MAIN[name], PASS_MAIN[name] = sent_scores, passage_scores
    print(name, {k: round(v, 2) for k, v in RES_MAIN[name].items()})

# ----------------------------------------------------------------------------------
# MQAG scoring rules (applied to the cached answer distributions)
# ----------------------------------------------------------------------------------
def mqag_question_score(item, n, method):
    from selfcheckgpt.modeling_selfcheck import method_simple_counting, method_vanilla_bayes, method_bayes_with_alpha
    prob, u = item["prob"], item["u"]
    prob_s, u_s = item["prob_s"][:n], item["u_s"][:n]
    k = len(u_s)
    if method == "counting":         return method_simple_counting(prob, u, prob_s, u_s, k, AT=0.5)
    if method == "bayes":            return method_vanilla_bayes(prob, u, prob_s, u_s, k, beta1=0.8, beta2=0.8, AT=0.5)
    if method == "bayes_with_alpha": return method_bayes_with_alpha(prob, u, prob_s, u_s, k, beta1=0.8, beta2=0.8)
    raise ValueError(method)

def mqag_scores(qcache, n=None, method="bayes_with_alpha"):
    # sentence score = mean over its questions; a sentence with no valid generated question gets the uninformative score 0.5
    out, empty = {}, 0
    for pid, per_sentence in qcache.items():
        sc = []
        for items in per_sentence:
            if not items: sc.append(0.5); empty += 1; continue
            sc.append(float(np.mean([mqag_question_score(it, n if n is not None else 10**9, method) for it in items])))
        out[pid] = np.array(sc)
    return out, empty