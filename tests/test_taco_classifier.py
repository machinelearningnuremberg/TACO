from sklearn.base import clone

from taco.model.tabpfn_arch.taco_classifier import TACOClassifier


def test_taco_classifier_supports_sklearn_parameter_protocol() -> None:
    classifier = TACOClassifier(
        device="cpu",
        n_estimators=3,
        use_compressor=True,
        wrapper_kwargs={},
    )

    params = classifier.get_params(deep=False)
    assert params["n_estimators"] == 3
    assert params["use_compressor"] is True

    cloned = clone(classifier)
    assert cloned.n_estimators == 3
    assert cloned.use_compressor is True

    classifier.set_params(n_estimators=2, use_compressor=False)
    assert classifier.n_estimators == 2
    assert classifier.use_compressor is False
