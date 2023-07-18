#  by yhpark 2023-07-10
from utils import *
import json
import cv2
from PIL import Image
import torch_pruning as tp

torch.backends.cudnn.benchmark = True  # 최적 cuda algorithm으로 변경 가능
def main():
    #set_random_seeds()
    device = device_check()

    # 0. dataset
    transform_ = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    val_dataset = datasets.ImageFolder('H:/dataset/imagenet100/val', transform=transform_)
    # img = cv2.imread('./data/test/ILSVRC2012_val_00023081.JPEG')    # image file load
    # cv2.imshow('tt', img)
    # cv2.waitKey(0)

    test_path = 'H:/dataset/imagenet100/val/n02077923/ILSVRC2012_val_00023081.JPEG'
    img = Image.open(test_path)
    #img.show()
    img_ = transform_(img).unsqueeze(dim=0).to(device)

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_count = len(classes)

    json_file = open('H:/dataset/imagenet100/Labels.json')
    class_name = json.load(json_file)
    # for i, v in enumerate(classes):
    #     print(f'{v} idx : {class_to_idx.get(v)}, name : {class_names.get(v)}')

    model_path0 = './checkpoints/resnet18.pth.tar'
    model_path1 = './checkpoints/resnet18_prun_loc_0.2.pth.tar'
    model_path2 = './checkpoints/resnet18_prun_loc_0.2_0.1.pth.tar'
    # print(f" model : {model_path}")
    model_paths = []
    model_paths.append(model_path1)


    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    for m_idx in range(len(model_paths)):
        print("=" * 50)
        model = torch.load(model_paths[m_idx], map_location=device).eval()
        if m_idx == 0:
            base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))

        # 3. inference
        for _ in range(100):
            out = model(img_)
        torch.cuda.synchronize()

        dur_time = 0
        iteration = 10000
        for _ in range(iteration):
            begin = time.time()
            out = model(img_)
            torch.cuda.synchronize()
            dur = time.time() - begin
            dur_time += dur
            #print(f'{i} dur time : {dur}')

        del model

        # 4. results
        print(f'{iteration}th iteration time : {dur_time} [sec]')
        print(f'Average fps : {1/(dur_time/iteration)} [fps]')
        print(f'Average inference time : {(dur_time/iteration)*1000} [msec]')
        max_tensor = out.max(dim=1)
        max_value = max_tensor[0].cpu().data.numpy()[0]
        max_index = max_tensor[1].cpu().data.numpy()[0]
        print(f'[{m_idx}] Resnet18 max index : {max_index}, value : {max_value}, class name : {classes[max_index]} {class_name.get(classes[max_index])}')



if __name__ == '__main__':
    main()
