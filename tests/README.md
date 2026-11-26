Run `python -m pytest -q`. The real-model test is skipped unless a model id is given:

    GIB_HF_TEST_MODEL=llava-hf/llava-interleave-qwen-0.5b-hf python -m pytest -q tests/test_hf_adapter.py
