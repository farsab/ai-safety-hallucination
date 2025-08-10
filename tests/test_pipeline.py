from src.pipeline import SafetyPipeline

def test_pipeline_instantiation():
    pipe = SafetyPipeline()
    assert pipe is not None
