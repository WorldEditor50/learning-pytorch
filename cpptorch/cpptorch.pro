TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

HEADERS += \
    mlp.h \
    net_op.h \
    tensor_op.h
SOURCES += \
        main.cpp

TORCHPATH = /home/eigen/MySpace/libtorch-cxx11-abi-shared-with-deps-1.10.1+cpu/libtorch
INCLUDEPATH += $$TORCHPATH/include
INCLUDEPATH += $$TORCHPATH/include/torch/csrc/api/include
LIBS += -L$$TORCHPATH/lib   -lasmjit \
                            -lgtest \
                            -lbackend_with_compiler \
                            -lgtest_main \
                            -lbenchmark \
                            -ljitbackend_test \
                            -lbenchmark_main \
                            -lkineto \
                            -lbreakpad \
                            -lmkldnn \
                            -lbreakpad_common \
                            -lnnapi_backend \
                            -lc10 \
                            -lnnpack \
                            -lcaffe2_detectron_ops \
                            -lnnpack_reference_layers \
                            -lcaffe2_module_test_dynamic \
                            -lonnx \
                            -lcaffe2_observers \
                            -lonnx_proto \
                            -lCaffe2_perfkernels_avx2 \
                            -lprotobuf \
                            -lCaffe2_perfkernels_avx512 \
                            -lprotobuf-lite \
                            -lCaffe2_perfkernels_avx \
                            -lprotoc \
                            -lcaffe2_protos \
                            -lpthreadpool \
                            -lclog \
                            -lpytorch_qnnpack \
                            -lcpuinfo \
                            -lqnnpack \
                            -lcpuinfo_internals \
                            -lshm \
                            -ldnnl \
                            -ltensorpipe \
                            -lfbgemm \
                            -ltensorpipe_uv \
                            -lfmt \
                            -ltorchbind_test \
                            -lfoxi_loader \
                            -ltorch_cpu \
                            -lgloo \
                            -ltorch_global_deps \
                            -lgmock \
                            -ltorch_python \
                            -lgmock_main \
                            -ltorch \
                            -lXNNPACK \
                            -lpthread

