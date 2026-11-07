"""
测试DDP与TensorDict的兼容性

验证三个关键场景：
1. 模型输入是TensorDict
2. 模型输出是TensorDict
3. Loss计算和梯度反向传播
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tensordict import TensorDict


class SimpleTensorDictModel(nn.Module):
    """接收和返回TensorDict的简单模型"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """
        输入：TensorDict with keys ['observation']
        输出：TensorDict with keys ['output', 'auxiliary']
        """
        x = td["observation"]
        output = self.net(x)

        # 返回TensorDict（模拟TorchRL的loss_module行为）
        # 使用batch_size=[]表示标量输出（类似SACLoss的返回）
        return TensorDict({
            "output": output,
            "auxiliary_scalar": output.mean(),  # 标量
        }, batch_size=[])


def setup_ddp(rank, world_size):
    """初始化DDP进程组"""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    """清理DDP进程组"""
    dist.destroy_process_group()


def test_ddp_tensordict(rank, world_size):
    """测试DDP与TensorDict的兼容性"""
    print(f"[Rank {rank}] 开始测试...")

    # 1. 初始化DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cpu")  # 使用CPU避免GPU依赖

    # 2. 创建模型并包装DDP
    model = SimpleTensorDictModel().to(device)
    ddp_model = DDP(model, device_ids=None)  # CPU模式不需要device_ids

    # 3. 创建优化器
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    # 4. 创建TensorDict输入（模拟batch数据）
    batch_size = 4
    input_td = TensorDict({
        "observation": torch.randn(batch_size, 10, device=device),
        "target": torch.randn(batch_size, 5, device=device),
    }, batch_size=batch_size)

    # 5. Forward pass（输入和输出都是TensorDict）
    output_td = ddp_model(input_td)

    print(f"[Rank {rank}] Forward成功！输出类型: {type(output_td)}")
    print(f"[Rank {rank}] 输出键: {list(output_td.keys())}")

    # 6. 计算Loss（使用TensorDict中的tensor）
    loss = ((output_td["output"] - input_td["target"]) ** 2).mean()
    print(f"[Rank {rank}] Loss计算成功: {loss.item():.4f}")

    # 7. Backward（验证梯度同步）
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度是否存在
    grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), float('inf'))
    print(f"[Rank {rank}] 梯度范数: {grad_norm:.4f}")

    # 8. Optimizer step
    optimizer.step()

    print(f"[Rank {rank}] ✅ 所有测试通过！")

    # 9. 清理
    cleanup_ddp()


def test_single_process():
    """单进程测试（验证基础功能）"""
    print("\n" + "="*60)
    print("🧪 单进程测试：验证TensorDict基础功能")
    print("="*60)

    model = SimpleTensorDictModel()

    # 创建输入
    input_td = TensorDict({
        "observation": torch.randn(4, 10),
        "target": torch.randn(4, 5),
    }, batch_size=4)

    # Forward
    output_td = model(input_td)

    # Loss和backward
    loss = ((output_td["output"] - input_td["target"]) ** 2).mean()
    loss.backward()

    print(f"✅ 单进程测试通过！")
    print(f"   - 输入类型: {type(input_td)}")
    print(f"   - 输出类型: {type(output_td)}")
    print(f"   - Loss: {loss.item():.4f}")


def test_torchrl_style_loss():
    """测试TorchRL风格的loss module（返回多个loss的TensorDict）"""
    print("\n" + "="*60)
    print("🧪 TorchRL风格Loss测试")
    print("="*60)

    class TorchRLStyleLoss(nn.Module):
        """模拟TorchRL的SACLoss返回格式"""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, td: TensorDict) -> TensorDict:
            output_td = self.model(td)

            # 返回多个loss（类似SAC的actor/critic/alpha loss）
            loss1 = ((output_td["output"] - td["target"]) ** 2).mean()
            loss2 = output_td["auxiliary_scalar"].abs()

            return TensorDict({
                "loss_primary": loss1,
                "loss_auxiliary": loss2,
                "total_loss": loss1 + loss2,
            }, batch_size=[])

    model = SimpleTensorDictModel()
    loss_module = TorchRLStyleLoss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 创建输入
    input_td = TensorDict({
        "observation": torch.randn(4, 10),
        "target": torch.randn(4, 5),
    }, batch_size=4)

    # Forward
    loss_td = loss_module(input_td)

    print(f"Loss TensorDict键: {list(loss_td.keys())}")
    print(f"  - loss_primary: {loss_td['loss_primary'].item():.4f}")
    print(f"  - loss_auxiliary: {loss_td['loss_auxiliary'].item():.4f}")
    print(f"  - total_loss: {loss_td['total_loss'].item():.4f}")

    # Backward（使用total_loss）
    optimizer.zero_grad()
    loss_td["total_loss"].backward()
    optimizer.step()

    print(f"✅ TorchRL风格Loss测试通过！")


if __name__ == "__main__":
    print("="*60)
    print("DDP与TensorDict兼容性测试套件")
    print("="*60)

    # 测试1：单进程基础功能
    test_single_process()

    # 测试2：TorchRL风格的loss module
    test_torchrl_style_loss()

    # 测试3：多进程DDP测试
    print("\n" + "="*60)
    print("🧪 多进程DDP测试")
    print("="*60)

    world_size = 2
    try:
        mp.spawn(
            test_ddp_tensordict,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("\n" + "="*60)
        print("🎉 所有测试通过！DDP与TensorDict完全兼容！")
        print("="*60)
    except Exception as e:
        print(f"\n❌ DDP测试失败: {e}")
        import traceback
        traceback.print_exc()
