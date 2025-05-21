# compile_pipeline.py
from kfp import compiler
from pipeline_resnet50 import defect_detection_pipeline

compiler.Compiler().compile(
    pipeline_func=defect_detection_pipeline,
    package_path="pipeline_visao.yaml"
)
