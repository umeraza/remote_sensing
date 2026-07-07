from geomex_vlm.data.schema import GeoInstructionSample


def test_schema_roundtrip():
    item = {"image":"x.png", "instruction":"Count aircraft", "answer":"2", "task":"counting", "class_id":1}
    sample = GeoInstructionSample.from_dict(item)
    assert sample.to_dict()["task"] == "counting"
