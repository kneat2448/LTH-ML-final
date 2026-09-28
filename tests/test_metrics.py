import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.metrics import classification_metrics, overlap_ratio


def test_metrics_match_sklearn():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 10, (500,), generator=g)
    logits = torch.randn(500, 10, generator=g) + 2.0 * torch.nn.functional.one_hot(y, 10)
    logits[:50] = logits[:50].round()  # ties in the AUC ranks
    m = classification_metrics(logits, y)
    yt, prob = y.numpy(), torch.softmax(logits.double(), 1).numpy()
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(yt, pred, average="macro", zero_division=0)
    assert np.isclose(m["acc"], accuracy_score(yt, pred))
    assert np.isclose(m["precision"], p) and np.isclose(m["recall"], r) and np.isclose(m["f1"], f1)
    assert np.isclose(m["auc"], roc_auc_score(yt, prob, multi_class="ovr", average="macro"))
    assert (np.array(m["confusion"]) == confusion_matrix(yt, pred, labels=range(10))).all()


def test_metrics_with_unpredicted_class():
    y = torch.tensor([0, 1, 1, 2, 2, 3])
    pred = [0, 1, 2, 2, 2, 2]  # class 3 never predicted
    logits = torch.nn.functional.one_hot(torch.tensor(pred), 4).float()
    m = classification_metrics(logits, y)
    p, r, f1, _ = precision_recall_fscore_support(y.numpy(), pred, average="macro", zero_division=0)
    assert np.isclose(m["precision"], p) and np.isclose(m["recall"], r) and np.isclose(m["f1"], f1)


def test_overlap_ratio_extremes():
    torch.manual_seed(0)
    w0 = {"w": torch.randn(1000)}
    mask = {"w": torch.ones(1000)}
    assert overlap_ratio(w0, w0, mask, 0.2) == 1.0
    rand = [overlap_ratio(w0, {"w": torch.randn(1000)}, mask, 0.2) for _ in range(20)]
    assert abs(np.mean(rand) - 0.2) < 0.03  # chance level R_p ~ p
