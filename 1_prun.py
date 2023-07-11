from prun_utils import *
import onnx
def main() :

    # model
    weight_path = 'model/vgg.cifar.pretrained.pth'
    checkpoint = torch.load(weight_path, map_location="cpu")
    model = VGG().cuda()
    print(f"=> loading checkpoint '{weight_path}'")
    model.load_state_dict(checkpoint['state_dict'])
    recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])

    # onnx model path
    export_model_path = f"./model/vgg_cifar.onnx"

    # 1. export onnx model
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).cuda()

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
    print("ONNX Model check done!")

    # dataset
    image_size = 32
    transforms = {"train": Compose([RandomCrop(image_size, padding=4),RandomHorizontalFlip(), ToTensor(),]), "test": ToTensor(),}
    dataset = {}

    for split in ["train", "test"]:
      dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms[split],)

    dataloader = {}
    for split in ['train', 'test']:
      dataloader[split] = DataLoader(dataset[split], batch_size=512, shuffle=(split == 'train'), num_workers=0, pin_memory=True,)

    dense_model_accuracy = evaluate(model, dataloader['test'])
    dense_model_size = get_model_size(model)
    print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
    print(f"dense model has size={dense_model_size/MiB:.2f} MiB")

    plot_weight_distribution(model)

    plot_num_parameters_distribution(model)

    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    pruned_model = channel_prune(model, prune_ratio=0.3)
    pruned_macs = get_model_macs(pruned_model, dummy_input)
    assert pruned_macs == 305388064
    print('* Check passed. Right MACs for the pruned model.')

    pruned_model_accuracy = evaluate(pruned_model, dataloader['test'])
    print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")

    print('Before sorting...')
    dense_model_accuracy = evaluate(model, dataloader['test'])
    print(f"dense model has accuracy={dense_model_accuracy:.2f}%")

    print('After sorting...')
    sorted_model = apply_channel_sorting(model)
    sorted_model_accuracy = evaluate(sorted_model, dataloader['test'])
    print(f"sorted model has accuracy={sorted_model_accuracy:.2f}%")

    # make sure accuracy does not change after sorting, since it is
    # equivalent transform
    assert abs(sorted_model_accuracy - dense_model_accuracy) < 0.1
    print('* Check passed.')

    channel_pruning_ratio = 0.3  # pruned-out ratio

    print(" * Without sorting...")
    pruned_model = channel_prune(model, channel_pruning_ratio)
    pruned_model_accuracy = evaluate(pruned_model, dataloader['test'])
    print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")


    print(" * With sorting...")
    sorted_model = apply_channel_sorting(model)
    pruned_model = channel_prune(sorted_model, channel_pruning_ratio)
    pruned_model_accuracy = evaluate(pruned_model, dataloader['test'])
    print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")

    num_finetune_epochs = 5
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(num_finetune_epochs):
        train(pruned_model, dataloader['train'], criterion, optimizer, scheduler)
        accuracy = evaluate(pruned_model, dataloader['test'])
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
        print(f'Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')



    table_template = "{:<15} {:<15} {:<15} {:<15}"
    print (table_template.format('', 'Original','Pruned','Reduction Ratio'))

    # 1. measure the latency of the original model and the pruned model on CPU
    #   which simulates inference on an edge device
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')
    pruned_model = pruned_model.to('cpu')
    model = model.to('cpu')

    pruned_latency = measure_latency(pruned_model, dummy_input)
    original_latency = measure_latency(model, dummy_input)
    print(table_template.format('Latency (ms)',
                                round(original_latency * 1000, 1),
                                round(pruned_latency * 1000, 1),
                                round(original_latency / pruned_latency, 1)))

    # 2. measure the computation (MACs)
    original_macs = get_model_macs(model, dummy_input)
    pruned_macs = get_model_macs(pruned_model, dummy_input)
    print(table_template.format('MACs (M)',
                                round(original_macs / 1e6),
                                round(pruned_macs / 1e6),
                                round(original_macs / pruned_macs, 1)))

    # 3. measure the model size (params)
    original_param = get_num_parameters(model)
    pruned_param = get_num_parameters(pruned_model)
    print(table_template.format('Param (M)',
                                round(original_param / 1e6, 2),
                                round(pruned_param / 1e6, 2),
                                round(original_param / pruned_param, 1)))

    # put model back to cuda
    pruned_model = pruned_model.to('cuda')
    model = model.to('cuda')



if __name__ == '__main__':
    main()



