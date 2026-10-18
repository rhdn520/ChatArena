#!/usr/bin/env python3
"""Initialize Arrow before vLLM native libraries, then run the API server.

Importing Arrow after vLLM crashed in libarrow's jemalloc background thread
in this runtime. The pre-submit gate uses this same import order.
"""
import pyarrow
import runpy

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
