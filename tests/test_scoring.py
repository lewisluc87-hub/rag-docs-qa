from scoring import RetrievalScore, false_positive_rate, summarize


def test_hit_true_when_any_expected_chunk_retrieved():
    s = RetrievalScore(expected_ids=["a", "b"], retrieved_ids=["x", "b", "y"])
    assert s.hit is True


def test_hit_false_when_no_expected_chunk_retrieved():
    s = RetrievalScore(expected_ids=["a", "b"], retrieved_ids=["x", "y"])
    assert s.hit is False


def test_adversarial_question_hit_is_false_not_true():
    # An adversarial question with empty expected_ids must never report
    # hit=True (vacuous truth from `any([])` would be wrong here --
    # "nothing expected, nothing retrieved" is not a successful retrieval).
    s = RetrievalScore(expected_ids=[], retrieved_ids=["x", "y"])
    assert s.hit is False
    assert s.is_adversarial is True


def test_recall_fraction_for_multi_expected_question():
    s = RetrievalScore(expected_ids=["a", "b", "c"], retrieved_ids=["a", "z", "c"])
    assert s.recall == 2 / 3


def test_recall_is_zero_for_adversarial_by_definition():
    s = RetrievalScore(expected_ids=[], retrieved_ids=["a", "b"])
    assert s.recall == 0.0


def test_recall_full_when_all_expected_found():
    s = RetrievalScore(expected_ids=["a"], retrieved_ids=["a", "b", "c"])
    assert s.recall == 1.0


def test_false_positive_rate_all_confident():
    assert false_positive_rate([0.1, 0.2, 0.3], threshold=0.5) == 1.0


def test_false_positive_rate_none_confident():
    assert false_positive_rate([0.9, 0.8], threshold=0.5) == 0.0


def test_false_positive_rate_empty_distances_is_zero_not_error():
    assert false_positive_rate([], threshold=0.5) == 0.0


def test_summarize_splits_answerable_and_adversarial_correctly():
    scores = [
        RetrievalScore(expected_ids=["a"], retrieved_ids=["a"]),  # hit
        RetrievalScore(expected_ids=["b"], retrieved_ids=["z"]),  # miss
        RetrievalScore(expected_ids=[], retrieved_ids=["q"]),  # adversarial
    ]
    summary = summarize(scores)
    assert summary["n_answerable"] == 2
    assert summary["n_adversarial"] == 1
    assert summary["hit_rate"] == 0.5


def test_summarize_with_no_answerable_questions_does_not_divide_by_zero():
    scores = [RetrievalScore(expected_ids=[], retrieved_ids=["q"])]
    summary = summarize(scores)
    assert summary["hit_rate"] is None
    assert summary["mean_recall"] is None


def test_summarize_counts_full_and_zero_recall_questions():
    scores = [
        RetrievalScore(expected_ids=["a", "b"], retrieved_ids=["a", "b"]),  # full recall
        RetrievalScore(expected_ids=["c", "d"], retrieved_ids=["x", "y"]),  # zero recall
        RetrievalScore(expected_ids=["e", "f"], retrieved_ids=["e", "x"]),  # partial recall
    ]
    summary = summarize(scores)
    assert summary["n_answerable_full_recall"] == 1
    assert summary["n_answerable_zero_recall"] == 1
