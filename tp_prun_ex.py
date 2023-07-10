#  by yhpark 2023-07-08
from utils import *
import onnx
import copy
import torch_pruning as tp


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    data_dir = 'H:/dataset/imagenet100'  # dataset path
    batch_size = 256
    workers = 8

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
    class_count = len(classes)

    # 1. model
    model_name = 'resnet18'
    pretrained = False
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_name))
        model = models.__dict__[model_name](weights='DEFAULT').to(device)
    else:
        print("=> using no pre-trained model model '{}'".format(model_name))
        model = models.__dict__[model_name]().to(device)

    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    if model_name == 'resnet18':
        if class_count != model.fc.out_features:
            model.fc = nn.Linear(model.fc.in_features, class_count)
    elif model_name == 'efficientnet_b0':
        if class_count != model.classifier[1].out_features:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    model = model.to(device)

    # evaluate model status
    if False:
        print(f"model: {model}")  # print model structure
        summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size
        test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
        print(f"acc before model train : {test_acc1}")

    check_path = './checkpoints/model_best_resnet18.pth.tar'
    load_checkpoint(check_path, model, device)
    test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    print(f" Acc after model train : {test_acc1}")

    sparse_model = copy.deepcopy(model)  # prevent overwrite

    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(sparse_model, example_inputs)

    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph().build_dependency(sparse_model, example_inputs=example_inputs)

    # 2. Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
    pruning_idxs = [2, 6, 9]
    pruning_group = DG.get_pruning_group(sparse_model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs)

    # 3. prune all grouped layer that is coupled with model.conv1
    if DG.check_pruning_group(pruning_group):
        pruning_group.prune()

    print("After pruning:")
    print(sparse_model)

    print("Let's inspect the pruning group. The results will show how a pruning operation triggers (=>) another one.")
    print(pruning_group)

    all_groups = list(DG.get_all_groups())
    print("Number of Groups: %d" % len(all_groups))
    print("The last Group:", all_groups[-1])

    test_acc1 = test(val_loader, sparse_model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    print(f" Acc after model pruning : {test_acc1}")

    macs, nparams = tp.utils.count_ops_and_params(sparse_model, example_inputs)
    print("   Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
    print("   MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))

    sparse_model = copy.deepcopy(model)  # prevent overwrite


    # 0. importance criterion for parameter selections
    #imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
    imp = tp.importance.MagnitudeImportance(p=1)

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in sparse_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == class_count:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    # 2. Pruner initialization
    iterative_steps = 5  # You can prune your model to the target sparsity iteratively.
    max_ch_sparsity = 1.0
    ch_sparsity_dict = {}
    pruner = tp.pruner.MagnitudePruner(
        sparse_model,
        example_inputs,
        importance=imp,  # importance criterion for parameter selection
        iterative_steps=iterative_steps,  # the number of iterations to achieve target sparsity
        ch_sparsity=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=max_ch_sparsity,
        ignored_layers=ignored_layers,
        global_pruning=False,  # If False, a uniform sparsity will be assigned to different layers.
    )

    for i in range(iterative_steps):
        print("=" * 16)
        # 3. the pruner.step will remove some channels from the model with least importance
        pruner.step()

        # 4. Do whatever you like here, such as fintuning
        macs, nparams = tp.utils.count_ops_and_params(sparse_model, example_inputs)
        print(sparse_model)
        print(sparse_model(example_inputs).shape)
        print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
        print("  Iter %d/%d, MACs: %.2f G => %.2f G" % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9))
        test_acc1 = test(val_loader, sparse_model, device, class_to_idx, classes, class_acc=False, print_freq=10)
        print(f" Acc after model pruning [{i}] : {test_acc1}")
        # finetune your model here
        # finetune(model)
        # ...


if __name__ == '__main__':
    main()
