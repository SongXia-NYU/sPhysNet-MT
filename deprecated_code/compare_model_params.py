import torch

if __name__ == '__main__':
    model1_path = "../exp297-uni_task_ss_run_2021-07-22_171405/best_model.pt"
    model2_path = "../exp325_run_2021-07-26_191533/best_model.pt"

    model1 = torch.load(model1_path)
    model2 = torch.load(model2_path)

    for key in model1.keys():
        assert key in model2.keys()
        diff_max = (model1[key] - model2[key]).abs().max()
        print(f"{key}: {diff_max}")
