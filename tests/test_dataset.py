from vlm_attr_extraction.data.dataset import read_jsonl

def test_train_file_exists():
    records = read_jsonl("data/sample/annotations/train.jsonl")
    assert len(records) > 0
