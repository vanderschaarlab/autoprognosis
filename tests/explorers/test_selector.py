# adjutorium absolute
from adjutorium.explorers.core.selector import PipelineSelector


def test_sanity() -> None:
    clf = PipelineSelector("lda")

    assert len(clf.imputers) == 0
    assert len(clf.feature_scaling) > 0

    assert clf.classifier.name() == "lda"
    assert clf.name() == "lda"

    assert len(clf.hyperparameter_space()) > 0
