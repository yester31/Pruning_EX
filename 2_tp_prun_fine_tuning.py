#  by yhpark 2023-07-09
# tensorboard --logdir ./logs
from utils import *
import onnx
import copy
import torch_pruning as tp
from functools import partial

genDir('./model')
genDir('./checkpoints')


def main():
    set_random_seeds()
    device = device_check()
    best_acc1 = 0

    # 0. dataset
    batch_size = 256
    workers = 8
    data_dir = 'H:/dataset/imagenet100'  # dataset path
    print(f"=> Custom {data_dir} is used!")
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(traindir, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    val_dataset = datasets.ImageFolder(valdir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True, sampler=None)

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    class_count = len(classes)

    # 1. model
    model_name = 'resnet18'
    pretrained = True
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
    if False:
        print(f"model: {model}")  # print model structure
        summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size

    # 2. load target pretrained model
    check_path = './checkpoints/model_best_resnet18.pth.tar'
    load_checkpoint(check_path, model, device)
    test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    print(f" Acc target model pruning : {test_acc1}")
    print(f"model: {model}")  # print model structure

    # 3. model pruning
    sparse_model = copy.deepcopy(model)  # prevent overwrite
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(sparse_model, example_inputs)

    # 3.1 ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in sparse_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == class_count:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    # 3.2. Pruner initialization
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean') # importance criterion
    iterative_steps = 1  # You can prune your model to the target sparsity iteratively.
    ch_sparsity = 0.2  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    max_ch_sparsity = 1.0
    global_pruning = False
    ch_sparsity_dict = {}
    pruner = tp.pruner.MagnitudePruner(
        sparse_model,
        example_inputs,
        importance=imp,  # importance criterion for parameter selection
        iterative_steps=iterative_steps,  # the number of iterations to achieve target sparsity
        ch_sparsity=ch_sparsity,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=max_ch_sparsity,
        ignored_layers=ignored_layers,
        global_pruning=global_pruning,  # If False, a uniform sparsity will be assigned to different layers.
    )

    print("=" * 16)
    # 3.3 the pruner.step will remove some channels from the model with least importance
    pruner.step()
    # 3.4 Do whatever you like here, such as fintuning
    macs, nparams = tp.utils.count_ops_and_params(sparse_model, example_inputs)
    print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
    print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
    # test_acc1 = test(val_loader, sparse_model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    # print(f" Acc of {save_name} [{i}] : {test_acc1}")
    print(sparse_model)


    # 4 finetune model
    if global_pruning:
        save_name = f'resnet18_prun_group_{str(ch_sparsity)}'
    else:
        save_name = f'resnet18_prun_loc_{str(ch_sparsity)}'
    print(f'==> {save_name}')
    test_acc1 = test(val_loader, sparse_model, device, class_to_idx, classes, class_acc=False, print_freq=20)
    print(f" Acc of {save_name} : {test_acc1}")

    writer = SummaryWriter(f'logs/{save_name}')
    epochs = 10
    learning_rate = 0.001

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(sparse_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(sparse_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(sparse_model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.4), int(epochs * 0.8)], gamma=0.1)

    print("=> Model training has started!")

    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, sparse_model, criterion, optimizer, epoch, device, scaler, use_amp, writer, None, 50)

        # evaluate on validation set
        acc1 = validate(val_loader, sparse_model, criterion, epoch * len(train_loader), device, class_to_idx, classes,
                        writer, False, 10)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_name,
            'state_dict': sparse_model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, save_name)

        if is_best:
            filename = f'./checkpoints/{model_name}.pth.tar'
            torch.save(sparse_model, filename)

    writer.close()

    # export onnx model
    onnx_flag = True
    if onnx_flag:
        export_model_path = f"./model/{save_name}.onnx"
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

        with torch.no_grad():
            torch.onnx.export(sparse_model,  # pytorch model
                              dummy_input,  # model dummy input
                              export_model_path,  # onnx model path
                              opset_version=17,  # the version of the opset
                              input_names=['input'],  # input name
                              output_names=['output'])  # output name

            print("ONNX Model exported at ", export_model_path)

        onnx_model = onnx.load(export_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX Model check done!")


if __name__ == '__main__':
    main()
