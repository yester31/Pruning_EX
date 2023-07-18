#  by yhpark 2023-07-10
import tensorrt as trt
import common
from utils import *
from PIL import Image
import json

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path="", precision="fp32"):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 30  # 29 : 512MiB, 30 : 1024MiB

            if precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    print("FP16 is not supported natively on this platform/device")
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "fp32":
                print("Using fp32 mode.")
            else:
                raise NotImplementedError(
                    f"Currently hasn't been implemented: {precision}."
                )

            if not os.path.exists(engine_file_path):
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print(
                        f"ONNX file {onnx_file_path} not found, please run 3_resnet18_onnx.py first to generate it."
                    )
                    exit(0)
                print("Loading ONNX file from path {}...".format(onnx_file_path))
                with open(onnx_file_path, "rb") as model:
                    print("Beginning ONNX file parsing")
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None

                network.get_input(0).shape = [1, 3, 224, 224]
                print("Completed parsing of ONNX file")
                print(
                    "Building an engine from file {}; this may take a while...".format(
                        onnx_file_path
                    )
                )
                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    engine_file_path = engine_file_path.replace(".trt", f"_{precision}.trt")
    print(engine_file_path)

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    set_random_seeds()
    device = device_check()
    dur_time = 0
    iteration = 10000

    # 2. input
    transform_ = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = datasets.ImageFolder(
        "H:/dataset/imagenet100/val", transform=transform_
    )

    test_path = "H:/dataset/imagenet100/val/n02077923/ILSVRC2012_val_00023081.JPEG"
    img = Image.open(test_path)
    img = transform_(img).unsqueeze(dim=0)
    num_img = np.array(img)
    x = np.array(num_img, dtype=np.float32, order="C")

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_count = len(classes)

    json_file = open("H:/dataset/imagenet100/Labels.json")
    class_name = json.load(json_file)

    # 3. tensorrt model
    model_name = "resnet18_imagenet_100"
    # model_name = 'resnet18_prun_loc_0.2'
    # model_name = 'resnet18_prun_loc_0.2_0.1'
    onnx_model_path = f"model/{model_name}.onnx"
    engine_file_path = f"model/{model_name}.trt"

    precision = "fp16"  # fp16 or fp32

    # Output shapes expected by the post-processor
    output_shapes = [(1, class_count)]

    # Do inference with TensorRT
    t_outputs = []
    with get_engine(
        onnx_model_path, engine_file_path, precision
    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = x
        for _ in range(100):
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        torch.cuda.synchronize()

        for i in range(iteration):
            begin = time.time()
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            torch.cuda.synchronize()
            dur = time.time() - begin
            dur_time += dur
        print("[TensorRT] {} iteration time : {} [sec]".format(iteration, dur_time))

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    t_outputs = [
        output.reshape(shape) for output, shape in zip(t_outputs, output_shapes)
    ]

    # 4. results
    print(f"{iteration}th iteration time : {dur_time} [sec]")
    print(f"Average fps : {1/(dur_time/iteration)} [fps]")
    print(f"Average inference time : {(dur_time/iteration)*1000} [msec]")
    max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
    max_value = max_tensor[0].cpu().data.numpy()[0]
    max_index = max_tensor[1].cpu().data.numpy()[0]
    print(
        f"Resnet18 max index : {max_index} , value : {max_value}, class name : {classes[max_index]} {class_name.get(classes[max_index])}"
    )


if __name__ == "__main__":
    main()
