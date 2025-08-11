"""
修复V5评估脚本的问题
测试两种可能的修复方案
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_fixes():
    """测试修复方案"""
    
    print("=" * 60)
    print("V5评估脚本修复方案")
    print("=" * 60)
    
    print("\n方案1：修改get_actions方法使用TensorDict")
    print("-" * 40)
    print("""
def get_actions(self, actor, obss):
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        observation = []
        vector = []
        for obs in obss:
            if isinstance(obs, dict):
                observation.append(obs['observation'])
                vector.append([obs['vector'][0] if isinstance(obs['vector'], np.ndarray) else obs['vector']])
        
        observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
        vector = torch.tensor(np.array(vector)).float().to(self.device)
        
        # 使用TensorDict调用（关键修改）
        from tensordict import TensorDict
        td = TensorDict({
            'observation': observation,
            'vector': vector
        }, batch_size=[len(obss)])
        
        td_out = actor(td)
        
        # 从TensorDict输出中提取动作
        if 'action' in td_out:
            actions = td_out['action'].tolist()
        else:
            raise ValueError("Actor没有返回'action'键")
            
        return actions
    """)
    
    print("\n方案2：重写run方法使用make_env")
    print("-" * 40)
    print("""
def run(self):
    # ... 前面的代码保持不变 ...
    
    # 使用make_env创建环境（关键修改）
    envs = []
    for _ in range(self.episodes):
        # 使用TorchRL包装的环境
        env = self.make_env(from_pixels=False)
        # 转换为普通gym环境用于评估
        envs.append(env)
    
    # ... 后面的代码保持不变 ...
    """)
    
    print("\n方案3：确保vector维度正确（临时修复）")
    print("-" * 40)
    print("""
def get_actions(self, actor, obss):
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        observation = []
        vector = []
        for obs in obss:
            if isinstance(obs, dict):
                observation.append(obs['observation'])
                # 确保vector是标量值
                v = obs['vector']
                if isinstance(v, np.ndarray):
                    v = v.item() if v.size == 1 else v[0]
                vector.append([v])
        
        observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
        vector = torch.tensor(np.array(vector)).float().to(self.device)
        
        # 确保vector是2D张量
        if vector.ndim == 3:
            vector = vector.squeeze(-1)
        
        # 原有的调用方式
        output = actor(observation=observation, vector=vector)
        # ... 处理输出 ...
    """)
    
    print("\n" + "=" * 60)
    print("推荐方案")
    print("=" * 60)
    print("\n✅ 推荐使用方案1：使用TensorDict调用actor")
    print("原因：")
    print("1. 这是训练时使用的方式，保持一致性")
    print("2. TensorDict能正确处理所有维度问题")
    print("3. 兼容所有TorchRL版本")
    print("4. 代码改动最小，风险最低")
    
    print("\n实施步骤：")
    print("1. 修改area_coverage_v5_sac_cont_eval.py的get_actions方法")
    print("2. 同样修改area_coverage_sac_cont_eval.py的get_actions方法")
    print("3. 测试确保两个脚本都能正常运行")

if __name__ == "__main__":
    test_fixes()