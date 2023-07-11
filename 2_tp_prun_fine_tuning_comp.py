#  by yhpark 2023-07-10
# tensorboard --logdir ./logs
from utils import *
import onnx
import copy
import torch_pruning as tp
from functools import partial
from prun_utils import *


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    batch_size = 256
    workers = 8
    data_dir = 'H:/dataset/imagenet100'  # dataset path
    print(f"=> Custom {data_dir} is used!")
    valdir = os.path.join(data_dir, 'val')

    val_dataset = datasets.ImageFolder(valdir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True, sampler=None)

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_acc = False # to check the accuracy of each class
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # 2. base model load
    check_path0 = './checkpoints/resnet18.pth.tar'
    print(f" model : {check_path0}")
    model = torch.load(check_path0, map_location=device)
    test(val_loader, model, device, class_to_idx, classes, class_acc=class_acc, print_freq=10)
    #print(f"model: {model}")  # print model structure
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # plot_weight_distribution(model)
    # plot_num_parameters_distribution(model)
    print("=" * 50)

    # 3. model pruning 1
    pruning_model_1_path = './checkpoints/resnet18_prun_loc_0.2.pth.tar'
    print(f" model : {pruning_model_1_path}")
    pruning_model_1 = torch.load(pruning_model_1_path, map_location=device)
    test(val_loader, pruning_model_1, device, class_to_idx, classes, class_acc=class_acc, print_freq=10)
    #print(f"model: {pruning_model_1}")  # print model structure
    pruning_model_1_macs, pruning_model_1_nparams = tp.utils.count_ops_and_params(pruning_model_1, example_inputs)
    print("[pruning 1 step] Params: %.2f M => %.2f M" % (base_nparams / 1e6, pruning_model_1_nparams / 1e6))
    print("[pruning 1 step] MACs: %.2f G => %.2f G" % (base_macs / 1e9, pruning_model_1_macs / 1e9))
    # plot_weight_distribution(pruning_model_1)
    # plot_num_parameters_distribution(pruning_model_1)
    print("=" * 50)


    # 4. model pruning 2
    pruning_model_2_path = f'./checkpoints/resnet18_prun_loc_0.2_0.1.pth.tar'
    print(f" model : {pruning_model_2_path}")
    pruning_model_2 = torch.load(pruning_model_2_path, map_location=device)
    test(val_loader, pruning_model_2, device, class_to_idx, classes, class_acc=class_acc, print_freq=10)
    #print(f"model: {pruning_model_2}")  # print model structure
    pruning_model_2_macs, pruning_model_2_nparams = tp.utils.count_ops_and_params(pruning_model_2, example_inputs)
    print("[pruning 2 step] Params: %.2f M => %.2f M => %.2f M" % (base_nparams / 1e6, pruning_model_1_nparams / 1e6, pruning_model_2_nparams / 1e6))
    print("[pruning 2 step] MACs: %.2f G => %.2f G => %.2f G" % (base_macs / 1e9, pruning_model_1_macs / 1e9, pruning_model_2_macs / 1e9))
    # plot_weight_distribution(pruning_model_2)
    # plot_num_parameters_distribution(pruning_model_2)
    print("=" * 50)

    # 5. export onnx model
    export_model_path = f"./model/resnet18_imagenet_100.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

    with torch.no_grad():
        torch.onnx.export(model,                      # pytorch model
                          dummy_input,              # model dummy input
                          export_model_path,        # onnx model path
                          opset_version=17,         # the version of the opset
                          input_names=['input'],    # input name
                          output_names=['output'])  # output name

        print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX Model check done! {export_model_path}")


    # 5. export onnx model
    export_model_path = f"./model/resnet18_prun_loc_0.2.onnx"

    with torch.no_grad():
        torch.onnx.export(pruning_model_1,                      # pytorch model
                          dummy_input,              # model dummy input
                          export_model_path,        # onnx model path
                          opset_version=17,         # the version of the opset
                          input_names=['input'],    # input name
                          output_names=['output'])  # output name

        print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX Model check done! {export_model_path}")


    # 5. export onnx model
    export_model_path = f"./model/resnet18_prun_loc_0.2_0.1.onnx"

    with torch.no_grad():
        torch.onnx.export(pruning_model_2,                      # pytorch model
                          dummy_input,              # model dummy input
                          export_model_path,        # onnx model path
                          opset_version=17,         # the version of the opset
                          input_names=['input'],    # input name
                          output_names=['output'])  # output name

        print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX Model check done! {export_model_path}")

if __name__ == '__main__':
    main()
