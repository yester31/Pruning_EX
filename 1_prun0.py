from utils import *
from prun_utils import *

def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    data_dir = 'H:/dataset/imagenet100'  # dataset path
    batch_size = 256
    workers = 8

    print(f"=> Custom {data_dir} is used!")
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(traindir, transform=Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    val_dataset = datasets.ImageFolder(valdir, transform=Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True, sampler=None)

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
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

    # 2. evaluate model
    #if False:
    print(f"model: {model}")  # print model structure
    summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size
    check_path = './checkpoints/model_best_resnet18.pth.tar'
    load_checkpoint(check_path, model, device)
    test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=True, print_freq=10)
    print(f"acc after model train : {test_acc1}")

    dense_model_accuracy = evaluate(model, val_loader)
    dense_model_size = get_model_size(model)
    print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
    print(f"dense model has size={dense_model_size / MiB:.2f} MiB")

    plot_weight_distribution(model)

    plot_num_parameters_distribution(model)

    # channel_pruning_ratio = 0.1  # pruned-out ratio

    # dummy_input = torch.randn(1, 3, 224, 224).cuda()
    # pruned_model = channel_prune_new(model, prune_ratio=channel_pruning_ratio)
    # pruned_macs = get_model_macs(pruned_model, dummy_input)
    # #assert pruned_macs == 305388064
    # print('* Check passed. Right MACs for the pruned model.')
    #
    # pruned_model_accuracy = evaluate(pruned_model, val_loader)
    # print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")
    #
    # print('Before sorting...')
    # dense_model_accuracy = evaluate(model, val_loader)
    # print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
    #
    # print('After sorting...')
    # sorted_model = apply_channel_sorting_new(model)
    # sorted_model_accuracy = evaluate(sorted_model, val_loader)
    # print(f"sorted model has accuracy={sorted_model_accuracy:.2f}%")
    #
    # # make sure accuracy does not change after sorting, since it is
    # # equivalent transform
    # assert abs(sorted_model_accuracy - dense_model_accuracy) < 0.1
    # print('* Check passed.')
    #
    #
    # print(" * Without sorting...")
    # pruned_model = channel_prune_new(model, channel_pruning_ratio)
    # pruned_model_accuracy = evaluate(pruned_model, val_loader)
    # print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")
    #
    # print(" * With sorting...")
    # sorted_model = apply_channel_sorting_new(model)
    # pruned_model = channel_prune_new(sorted_model, channel_pruning_ratio)
    # pruned_model_accuracy = evaluate(pruned_model, val_loader)
    # print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")
    #
    # num_finetune_epochs = 5
    # optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    # criterion = nn.CrossEntropyLoss()
    #
    # best_accuracy = 0
    # for epoch in range(num_finetune_epochs):
    #     train(pruned_model, train_loader, criterion, optimizer, scheduler)
    #     accuracy = evaluate(pruned_model, val_loader)
    #     is_best = accuracy > best_accuracy
    #     if is_best:
    #         best_accuracy = accuracy
    #     print(f'Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
    #
    # table_template = "{:<15} {:<15} {:<15} {:<15}"
    # print(table_template.format('', 'Original', 'Pruned', 'Reduction Ratio'))
    #
    # # 1. measure the latency of the original model and the pruned model on CPU
    # #   which simulates inference on an edge device
    # dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
    # pruned_model = pruned_model.to('cpu')
    # model = model.to('cpu')
    #
    # pruned_latency = measure_latency(pruned_model, dummy_input)
    # original_latency = measure_latency(model, dummy_input)
    # print(table_template.format('Latency (ms)',
    #                             round(original_latency * 1000, 1),
    #                             round(pruned_latency * 1000, 1),
    #                             round(original_latency / pruned_latency, 1)))
    #
    # # 2. measure the computation (MACs)
    # original_macs = get_model_macs(model, dummy_input)
    # pruned_macs = get_model_macs(pruned_model, dummy_input)
    # print(table_template.format('MACs (M)',
    #                             round(original_macs / 1e6),
    #                             round(pruned_macs / 1e6),
    #                             round(original_macs / pruned_macs, 1)))
    #
    # # 3. measure the model size (params)
    # original_param = get_num_parameters(model)
    # pruned_param = get_num_parameters(pruned_model)
    # print(table_template.format('Param (M)',
    #                             round(original_param / 1e6, 2),
    #                             round(pruned_param / 1e6, 2),
    #                             round(original_param / pruned_param, 1)))
    #
    # # put model back to cuda
    # pruned_model = pruned_model.to('cuda')
    # model = model.to('cuda')


if __name__ == '__main__':
    main()



