#  by yhpark 2023-07-07
from utils import *
import onnx

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
    if True:
        print(f"model: {model}")  # print model structure
        summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size

    # 2. evaluate model
    base_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    print(f"acc before model train : {base_acc1}")

    check_path = './checkpoints/model_best_resnet18.pth.tar'
    load_checkpoint(check_path, model, device)
    target_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=True, print_freq=10)
    print(f"acc after model train : {target_acc1}")

    # 3. export onnx model
    export_model_path = f"./model/resnet18_imagenet100.onnx"
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
    print("ONNX Model check done!")


if __name__ == '__main__':
    main()
