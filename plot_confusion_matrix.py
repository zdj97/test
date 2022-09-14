import os
import copy
import click
import torch
import torch.utils.data
import torch.distributed as dist
import bc_resnet_model
import get_data
import train
import apply
import util
import time
import json
from confusiontable import ConfusionMatrix
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
print(torch.__version__)
def run(model, train_loader, validation_loader, test_loader, optimizer, scheduler, device, checkpoint_file, n_epoch=10, log_interval=100):
    best_score = 0
    best_model = copy.deepcopy(model)
    for epoch in range(n_epoch):
        print(f"--- start epoch {epoch} ---")
        train.train_epoch(model, optimizer, train_loader, device, epoch, log_interval=log_interval)
        if scheduler:
            scheduler.step()
        score = apply.compute_accuracy(model, validation_loader, device)
        print(f"Validation accuracy: {score:.5f}")
        if best_score < score:
            best_score = score
            best_model = copy.deepcopy(model)
            torch.save(best_model.module.state_dict(), checkpoint_file)
    print(f"Top validation accuracy: {best_score:.5f}")
    test_score = apply.compute_accuracy(best_model, test_loader, device)
    print(f"Test accuracy: {test_score:.5f}")


@click.group(help="Train and apply BC-ResNet Keyword Spotting Model")
def cli():
    pass

@cli.command("test", help="Test model accuracy on test set")
@click.option("--model-file", type=str, help="path to model weights")
@click.option("--scale", type=int, default=1, help="model width will be multiplied by scale")
@click.option("--batch-size", type=int, default=64, help="batch size")
@click.option("--device", type=str, default=util.get_device(), help="`cuda` or `cpu`")
@click.option("--dropout", type=float, default=0.1, help="dropout")
@click.option("--subspectral-norm/--dropout-norm", type=bool, default=True, help="use SubspectralNorm or Dropout")
def test_command(model_file, scale, batch_size, device, dropout, subspectral_norm):
    if not os.path.exists(model_file):
        raise FileExistsError(f"model {model_file} not exists")

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print(f"Device: {device}")
    print(f"Use subspectral norm: {subspectral_norm}")

    model = bc_resnet_model.BcResNetModel(
        n_class=get_data.N_CLASS,
        scale=scale,
        dropout=dropout,
        use_subspectral=subspectral_norm,
    ).to(device)
    model.load_state_dict(torch.load(model_file))

    test_loader = torch.utils.data.DataLoader(
        get_data.SubsetSC(subset="testing"),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_data.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    ori_time=time.time()
    test_score,far,ffr = apply.compute_accuracy(model, test_loader, device)
    cur_time=time.time()

    print(f"Test accuracy: {test_score}, 误报率: {far}, 漏报率: {ffr}")
    
    print('测试运行时间为：{} s'.format(cur_time-ori_time))
    plot_ConMatrix(model,test_loader,device,'./class_indices.json',3)


from apply import get_likely_index
def plot_ConMatrix(model,test_loader,device,json_file,class_nums):
    model.eval()
    json_F = open(json_file, 'r')
    class_indict = json.load(json_F)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=class_nums, labels=labels)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            pred,out = model(data)
            pred=get_likely_index(pred,out)
            confusion.update(pred.to("cpu").numpy(), target.to("cpu").numpy())
    confusion.summary()


if __name__ == "__main__":
    cli()
