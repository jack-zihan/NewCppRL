思考最简洁、优雅、高效清晰的实现方式。

1.  在cpp_env_base.py中，更新配置功能不完善，更新配置只对奖励系统有效，其他的都还没有写。但是我不想每个组件都要写一个类似.update_coefficients的更新配置函数，感觉破坏了之前代码的有雅性和专注度，有没有什么方法可以用一种什么魔法方法或者直接把其中的self.config类拉取出来怎样更新，或者self.config是更新函数，在EnvironmentConfig有个什么更新函数之内的更新方法来更新成员变量？
    
    ```
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update environment configuration.
    
        Args:
            new_config: New configuration values
        """
        # Update reward coefficients if provided
        if 'reward_coefficients' in new_config:
            self.reward_system.update_coefficients(new_config['reward_coefficients'])
    
        # Additional config updates can be added here
    
    ```
    

2. 在cpp_env_base.py中的 to_dict()好像没有这个功能？

```
def get_state_info(self) -> Dict[str, Any]:
    """Get current environment state information."""
    return self.env_state.to_dict()
```

环境状态好像没有to_dict()这个功能？思考一下我为什么需要这个功能？这个功能是不是其实没有业务用途？直接删除还是return self.env_state就好呢，为什么需要to_dict()呢

```
def get_state_info(self) -> Dict[str, Any]:
    """Get current environment state information."""
    return self.env_state.to_dict()
```



3. 在observation.py中，感觉逻辑还是有问题，这个extract_ego_patch首先会进行padding扩充，目的是旋转 后crop不会越界，或者说旋转后一些原来没在地图上的区域应该在地图上了，这些没在地图上的区域应该有一个确定的赋值，所以先扩充再旋转，再crop尺寸就保证crop下来的区域值都是对的，这是操作的目的，但是先先扩充，然后再裁剪回去是为什么？帮助认真思考，深入分析一下需要裁剪回去吗？

```jsx
def extract_ego_patch(maps: np.ndarray, pad_values: List[float],
                     center_y: float, center_x: float, direction_deg: float,
                     patch_size: Tuple[int, int]) -> np.ndarray:
    """
    提取以智能体为中心的自我中心观察补丁
    
    Args:
        maps: 堆叠的地图 (H, W, C)
        pad_values: 每个通道的填充值
        center_y: 中心Y坐标
        center_x: 中心X坐标
        direction_deg: 方向角度
        patch_size: 输出补丁大小 (高度, 宽度)
        
    Returns:
        旋转和裁剪后的观察补丁
    """
    patch_height, patch_width = patch_size
    
    # 计算旋转所需的对角线填充长度
    diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
    
    # 应用填充
    padded_maps = apply_channel_padding(maps, pad_values, diagonal_length)
    
    # 调整中心坐标以适应填充
    center_y_padded = center_y + diagonal_length
    center_x_padded = center_x + diagonal_length
    
    # 围绕中心裁剪正方形区域
    top = int(round(center_y_padded - diagonal_length))
    bottom = int(round(center_y_padded + diagonal_length))
    left = int(round(center_x_padded - diagonal_length))
    right = int(round(center_x_padded + diagonal_length))
    
    cropped_maps = padded_maps[top:bottom, left:right, :]
    
    if cropped_maps.size == 0:
        raise ValueError("裁剪区域为空")
    
    # 旋转以使智能体方向向上对齐
    # 180度是因为要将agent的前进方向（默认向右0度）旋转到图像上方
    rotation_angle = 180 + direction_deg
    rotation_center = (diagonal_length, diagonal_length)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
    
    rotated_maps = cv2.warpAffine(
        cropped_maps,
        rotation_matrix,
        (cropped_maps.shape[1], cropped_maps.shape[0])
    )
    
    # 确保为3D
    if rotated_maps.ndim == 2:
        rotated_maps = rotated_maps[..., np.newaxis]
    
    # 最终裁剪到所需的补丁大小
    rotated_height, rotated_width = rotated_maps.shape[:2]
    start_y = max(0, (rotated_height - patch_height) // 2)
    start_x = max(0, (rotated_width - patch_width) // 2)
    
    final_patch = rotated_maps[start_y:start_y + patch_height, 
                             start_x:start_x + patch_width, :]
    
    return final_patch
```

裁剪回去不就没意义了吗

    patch_height, patch_width = patch_size
    
    # 计算旋转所需的对角线填充长度
    diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
    
    # 应用填充
    padded_maps = apply_channel_padding(maps, pad_values, diagonal_length)
    
    # 调整中心坐标以适应填充
    center_y_padded = center_y + diagonal_length
    center_x_padded = center_x + diagonal_length
    
    # 围绕中心裁剪正方形区域
    top = int(round(center_y_padded - diagonal_length))
    bottom = int(round(center_y_padded + diagonal_length))
    left = int(round(center_x_padded - diagonal_length))
    right = int(round(center_x_padded + diagonal_length))
    
    cropped_maps = padded_maps[top:bottom, left:right, :]

我觉得不应该是填充、旋转，然后crop回原尺寸，就完成了extract_ego_patch的任务了么，另外我还在思考，这样的操作是不是非常复杂，我们想要的这种效果是不是可以基于cv2.warpAffine自身的borderMode: int = ...,
borderValue...等等参数直接实现，而不用我自己再扩充填充等等操作了，不确定，但是不一样的是可能每个通道填充的值是不一样的，比如 weed_map和frontier我们希望填0，obstacle我们希望填充0...所以填充需要跟输入的padding_value对应（但是旧版扩充也是直接给mask实现的，opencv或者其他是不是有实现这种效果的方法        maps = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                  cv2.BORDER_CONSTANT, value=mask, )）。请基于CLAUDE.md的优雅、高效、清晰、简洁、以及Less is more的代码设计和实现原则，思考，这项功能应该怎么实现最简洁、优雅，但有清晰高效，思考最好、最合适的优化方案。

4. 边界填充策略中，你提到

```jsx
**边界填充策略**

- 旧：cv2.BORDER_CONSTANT统一填充
- 新：每个通道独立的pad值（更灵活）  

        maps = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                  cv2.BORDER_CONSTANT, value=mask, )
不也是这些值吗
```

但是我不确定是不是这样，旧版不也是使用

        maps = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                  cv2.BORDER_CONSTANT, value=mask, )

根据自己的值进行填充的吗，新旧真的有区别吗

5. 在获得观察以及渲染图片的时候，思考都出现了基于agent_direction旋转180°的操作（180 + agent_direction），再次验证一下新旧两版是否完全一致的，在旧版每个应该有的位置，新版都对应进行了旋转操作。

6. **障碍物APF后处理Bug  这个是有道理的，认可方案**

**Bug描述**：
新环境缺失了障碍物APF的关键后处理步骤！

```
# 旧环境 - 有后处理
apf_obstacle = np.maximum(apf_obstacle, np.logical_and(self.map_obstacle, self.map_mist))

# 新环境 - 缺失这行！
# 这会导致障碍物本身在APF场中的值可能不是1
```

**影响分析**：

- 障碍物中心的APF值可能小于1，影响避障行为
- 智能体对障碍物的感知可能减弱
- 训练出的策略可能有更高的碰撞风险

**修复方案**：

```
# 在envs_new/cpp_env_v2.py line 152后添加：
apf_obstacle = np.maximum(apf_obstacle, np.logical_and(map_obstacle, map_mist))
```

回复：我认为你说的有道理，这个地方需要解决，请着手解决一下



7. **噪声注入机制差异**

### **Bug严重度：High**

**问题定位**：噪声应用的时机和方式

**详细对比**：

```
# 旧环境：在get_rotated_obs_中直接修改
def get_rotated_obs_(self, maps, mask):
    agent_y = self.agent.y
    agent_x = self.agent.x
    if self.noise_position:
        delta_y = np.clip(self.np_random.normal(0, self.noise_position), ...)
        agent_y += delta_y
        agent_x += delta_x

# 新环境：通过apply_noise_to_pose函数
noisy_y, noisy_x, noisy_direction = apply_noise_to_pose(
    agent.y, agent.x, agent.direction,
    self.config.position_noise, self.config.direction_noise,
    self.rng or np.random.default_rng()
)
```

**潜在问题**：

1. **随机数生成器不一致**：旧版用`self.np_random`，新版用`self.rng or np.random.default_rng()`
2. **噪声缓存问题**：如果多次调用，噪声值是否一致？
3. **clip范围可能不同**：需要验证apply_noise_to_pose的实现

**修复建议**：

```
# 确保使用相同的随机数生成器
self.rng = self.np_random  # 在ObservationGenerator初始化时设置
```

回复：为什么还要self.rng or np.random.default_rng()的or np.random.default_rng()呢，不是直接传入self.rng就行了吗，另外要深入取分析这个rng是怎么使用的

```
def set_random_generator(self, rng: np.random.Generator) -> None:
    self.rng = rng

```

新版和旧版创造的噪声分布以及分布参数是不是一样的（也就是这些位置朝向采样是不是都在一个空间），如果是一样的，那其实就没关系，请仔细检查确认一下：

```jsx
def apply_noise_to_pose(y: float, x: float, direction: float,
                       position_noise: float, direction_noise: float,
                       rng: np.random.Generator) -> Tuple[float, float, float]:
    """对智能体姿态应用噪声"""
    if position_noise > 0:
        y_noise = rng.normal(0, position_noise)
        x_noise = rng.normal(0, position_noise)
        y += np.clip(y_noise, -position_noise, position_noise)
        x += np.clip(x_noise, -position_noise, position_noise)
    
    if direction_noise > 0:
        dir_noise = rng.normal(0, direction_noise)
        direction = (direction + np.clip(dir_noise, -direction_noise, direction_noise)) % 360
    
    return y, x, direction

```

应用噪声的时候可以or np.random.default_rng()是不是都可以统一去掉

8. agent_direction应该是角度制的（角速度命令也是角度制的），检查一下于此有关的计算和渲染的时候是否是正确的，没有吧角度制的一些操作用与弧度制搞混吧。



9. **随机数生成器状态管理Bug 🚨**

### **Bug严重度：Critical**

**问题定位**：随机数生成器的初始化和管理

**诊断发现**：

```jsx
# 旧环境：统一使用self.np_random
self.np_random.normal(0, self.noise_position)
self.np_random.uniform() < self.noise_weed

# 新环境：混合使用
self.np_random.uniform() < self.noise_weed  # 在cpp_env_v2.py
self.rng or np.random.default_rng()  # 在observation_generator.py
```

**影响**：

- 即使设置相同种子，噪声序列也会不同
- 导致观测生成的不可复现性
- 破坏了确定性测试的基础

**修复方案**：

```
# 在reset时同步所有随机数生成器
def reset(self, seed=None):
    super().reset(seed=seed)
    self.scenario_generator.set_random_generator(self.np_random)
    self.observation_generator.set_random_generator(self.np_random)
    # 确保所有组件使用相同的RNG
```

回复：我认为这个问题也是or np.random.default_rng()造成的，帮我思考一下去掉这个是不是就正常修复了。

10. 多尺度观测实现差异

### **Bug严重度：High**

**问题定位**：多尺度变换的中心裁剪计算

**关键差异**：

```
# 旧环境：简单的中心裁剪
center_size = self.state_downsize[0] // 2
obs_list.append(obs_[:,
    (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
    (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2)
])

# 新环境：考虑了边界情况
if scale == 3 and cropped.shape[1] < feature_size:
    # 需要resize处理
    cropped_tensor = torch.from_numpy(cropped).unsqueeze(0)
    resized = F.interpolate(cropped_tensor, size=(feature_size, feature_size), mode='nearest')
```

**潜在问题**：

- 第4层（scale=3）的处理逻辑不同
- resize操作可能引入数值差异
- 边界处理策略不一致

回复：你说的这个我认为确实有一些问题，我们的多尺度化的思路其实是这样：首先在之前的操作中我们已经将地图旋转平移到了agent的局部坐标系（agent的位置为图像的中心，agent的朝向为图片正向朝上），多尺度化即是以agent位姿（图片中心）为基准，crop多个尺度的图片下来（比如16 * 16, 32*32, 64*64, 128*128），代表不同的视野尺度，然后我们再将多张图片resize回16*16（只是举例，这些值是参数设置的）大小，形成小尺度精细，大尺度粗糙（128*128  → 16 * 16）的布局，这也符合人类的感知直觉，近处精细，远处有个粗略的感知即可。但是呢，如果用简单的resize或者平均池化来放缩图像尺寸是有问题的，这些图片都是0-1的掩码，比如weed_map更是一些零星的像素点，resize以及平均池化再量化的过程中很可能丢失这些目标（1被周围的分散就成0了），因此在原环境中使用的是最大池化进行图片尺寸的归一化（F.max_pool2d），确保1的位置永远都是1。在实现的时候，思路稍稍做了微调：

```jsx
    def get_sgcnn_obs(self, obs: np.ndarray,
                      maps: Optional[np.ndarray] = None,
                      mask: Optional[Sequence[float]] = None):  # 在obs的基础上做多尺度图
        obs_ = obs
        obs_list = []
        center_size = self.state_downsize[0] // 2
        with torch.no_grad():
            for _ in range(4): #  从最小一层obs到最大一层obs，每次最大池化一次，然后crop sgcnn_size
                obs_list.append(obs_[
                                :,
                                (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                                (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                                ])
                obs_ = F.max_pool2d(torch.from_numpy(obs_), (2, 2), 2).numpy()
                center_size //= 2
            if self.use_global_obs:
                obs_global = self.get_global_obs(maps, mask)
                obs_global = obs_global.transpose(2, 0, 1)
                obs_list.append(obs_global)
        return np.concatenate(obs_list, axis=0, dtype=np.float32)
```

先crop self.sgcnn_size尺寸的图片，这是最精细的图片，然后通过最大池化将图片尺寸缩小为二分之一，然后crop self.sgcnn_size尺寸同步，这样的效果等效于相当于crop了 2*self.sgcnn_size, 2* self.sgcnn_size的图像最大池化回去，只是以更简洁优雅的方式实现的。

新版代码的实现方式和效果我不确定有没有问题，这个应该是重点，你帮我认真思考，深入核对一下，感觉实现的有点奇奇怪怪的，先是scale=0的时候直接crop，然后后续先最大池化，然后crop, 感觉实现上没有之前那么清晰、优雅、简洁。最后==3的时候进行尺寸验证感觉是不需要的：

```jsx
                    if scale == 3 and cropped.shape[1] < feature_size:
                        cropped_tensor = torch.from_numpy(cropped).unsqueeze(0)
                        resized = F.interpolate(
                            cropped_tensor, 
                            size=(feature_size, feature_size), 
                            mode='nearest'
                        )
                        cropped = resized.squeeze(0).numpy()
```

我们应该在最开始进行多尺度化之前就验证清楚进行多轮多尺度的尺寸是否是有效的，比如判断crop的最大尺寸是否会超出当前的图片尺寸，不会才继续操作，如何超出了应该报错，这这种情况应该是认为参数设计有问题，而不是通过而不是通过插值掩盖问题。

11. **全局观测噪声应用Bug**

### **Bug严重度：Medium**

**问题定位**：全局观测的噪声处理

**诊断发现**：
新环境在全局观测时重复计算了噪声：

```
# 新环境 - 重复计算噪声
if self.config.use_global_features:
    noisy_y, noisy_x, noisy_direction = apply_noise_to_pose(...)  # 第二次计算！
```

**影响**：

- 全局观测和局部观测使用不同的噪声值
- 可能导致观测不一致

**修复建议**：

```
# 缓存噪声值，避免重复计算
def _extract_base_observation(self, ...):
    # 计算一次，缓存结果
    self._cached_noisy_pose = apply_noise_to_pose(...)
    # 后续使用缓存值
```

回复：关于这个问题，我认为也是存在的，而且现在的操作是不合理的，添加全局观测的目的是多尺度的时候是由agent中心crop的，这样可能导致丢失最大crop尺度外的全局信息，因此希望将旋转平移后的agent坐标系的全局地图也resize会多尺度的图片的尺寸，concat在原来的多尺度地图进去，所以，看着下面添加全局观测的问题：

```jsx
            # 如果启用，添加全局观察
            if self.config.use_global_features:
                # 全局观测需要包含噪声，复用_extract_base_observation的噪声计算
                noisy_y, noisy_x, noisy_direction = apply_noise_to_pose(
                    agent.y, agent.x, agent.direction,
                    self.config.position_noise, self.config.direction_noise,
                    self.rng or np.random.default_rng()
                )
                
                # 使用完整地图尺寸提取全局视角
                global_observation = extract_ego_patch(
                    maps=stacked_maps,
                    pad_values=pad_values,
                    center_y=noisy_y,
                    center_x=noisy_x,
                    direction_deg=noisy_direction,
                    patch_size=(stacked_maps.shape[0], stacked_maps.shape[1])
                )
                
                # Resize到feature_size
                global_resized = cv2.resize(
                    global_observation,
                    (feature_size, feature_size),
                    interpolation=cv2.INTER_AREA
                )
                
                # 转换为 (C, H, W) 格式
                obs_global = global_resized.transpose(2, 0, 1)
                obs_list.append(obs_global)
        
```

首先，感觉是不需要再计算一遍extract_ego_patch了，之前在

```jsx
    def generate_observation(self, agent: Agent, maps_input: Union[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]) -> np.ndarray:
        """
        生成观察
        
        Args:
            agent: 智能体
            maps_input: 地图输入，可以是：
                       - Dict[str, Dict[str, Any]]: 包含map和pad信息的字典（旧格式）
                       - Dict[str, np.ndarray]: 直接的地图数组字典（新格式）
            
        Returns:
            观察数组，格式为 (C, H, W)
        """
        # 1. 预处理地图
        maps_dict = self._preprocess_maps(maps_input)
        stacked_maps, pad_values = stack_maps(maps_dict)
        
        # 2. 生成基础ego-centric观测（包含噪声）
        base_observation = self._extract_base_observation(agent, stacked_maps, pad_values)
        
        # 3. 根据配置进行不同的处理
        if self.config.use_multiscale:
            return self._apply_multiscale_transform(base_observation, agent, stacked_maps, pad_values)
        else:
            return base_observation
```

的时候已经生成了局部坐标系的base_observation, 也就是输入self._apply_multiscale_transform的obs_current = base_observation.copy()已经就是全局观察了，不再需要自己再添加噪声计算一遍，当前的写法是错误的，另外也不是Resize到feature_size到原尺寸，之前说过了应该是最大池化回去，否则就没有效果了，回去一下旧环境的get_global_obs_，其实是用的最大池化

```jsx
    def get_global_obs_(self, maps, mask: Sequence[float]):
        agent_y = self.agent.y
        agent_x = self.agent.x
        if self.noise_position:
            delta_y = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            delta_x = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            agent_y += delta_y
            agent_x += delta_x
        agent_direction = self.agent.direction
        if self.noise_direction:
            delta_direction = np.clip(self.np_random.normal(0, self.noise_direction),
                                      -self.noise_direction,
                                      self.noise_direction)
            agent_direction += delta_direction
            agent_direction %= 360

        diag_r = self.dimensions[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs_global = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                        cv2.BORDER_CONSTANT, value=mask, )
        leftmost = round(agent_y)
        rightmost = round(agent_y + 2 * diag_r_int)
        upmost = round(agent_x)
        bottommost = round(agent_x + 2 * diag_r_int)
        obs_cropped = obs_global[leftmost:rightmost, upmost:bottommost]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.dimensions[0] / 2)
        delta_rightmost = delta_leftmost + self.dimensions[0]
        obs_global = cv2.warpAffine(obs_cropped.astype(np.float32), rotation_mat, (dst_size, dst_size))
        obs_global = obs_global[delta_leftmost:delta_rightmost, delta_leftmost:delta_rightmost]
        if obs_global.ndim == 2:
            obs_global = obs_global.reshape(*obs_global.shape, 1)
        obs_global = obs_global.transpose(2, 0, 1)
        kernel_size = int(np.round(self.dimensions[0] / self.sgcnn_size)) - 1
        obs_global = F.max_pool2d(torch.from_numpy(obs_global),
                                  (kernel_size, kernel_size),
                                  kernel_size).numpy()
        obs_global = obs_global.transpose(1, 2, 0)
        return obs_global
```

但是原版虽然功能正确，但是代码非常冗余，不优雅，把旋转到局部又做了一遍，另外，其噪声也重新计算了一次而不是用之前噪声计算全局地图，这点其实旧版就是不对的，我们在新版应该扬长避短，即确保实现方式优雅，又保证这些思考的地方是正确的。

12. **渲染层次顺序差异**

### **Bug严重度：Medium**

**问题定位**：渲染顺序的变化

**详细对比**：

```
旧环境渲染顺序：
1. background (白色)
2. field_frontier
3. covered_farmland
4. agent_vision (椭圆) ← 位置4
5. weeds (enlarged)
6. trajectory
7. obstacles ← 位置7
8. agent
9. mist effect

新环境渲染顺序：
1. background
2. field_frontier
3. covered_farmland
4. obstacles ← 位置4（提前了）
5. agent_vision ← 位置5（延后了）
6. weeds
7. trajectory
8. agent
9. mist effect
```

**影响分析**：

- obstacles会被agent_vision覆盖（新版本）
- agent_vision会被obstacles覆盖（旧版本）
- 视觉效果会有差异，但不影响功能

**修复建议**：
保持与旧版本一致的渲染顺序。

回复：好的，可以将4，5调换一下，保持和旧版一致。

13. **透明度值不一致**

### **Bug严重度：Medium**

**问题定位**：Mist效果的透明度值

**差异**：

- 旧环境：0.7
- 新环境：MIST_EFFECT_ALPHA = 0.7（一致）

但是covered_weed的透明度有差异：

- 旧环境：0.1混合系数
- 新环境：COVERED_WEED_ALPHA = 0.1（但应用方式相反）

```
# 旧环境
0.9 * np.array((0, 0, 0)) + 0.1 * rendered_map

# 新环境
(1 - COVERED_WEED_ALPHA) * np.array(RENDER_COLORS['weed_covered']) +
COVERED_WEED_ALPHA * rendered_map[weed_covered]
```

**修复**：调整透明度应用方式保持一致。

这个没有特别听懂，再深入讲解分析一样，其渲染出来的效果区别

14. **第一人称视图提取差异**

### **Bug严重度：High**

**问题定位**：第一人称视图的提取方法

**诊断发现**：

- 旧环境：使用render_self()方法，先渲染再提取
- 新环境：使用extract_ego_patch()函数，直接从渲染地图提取

**潜在问题**：

1. 边界填充值可能不同
2. 旋转插值方法可能不同
3. 裁剪精度可能有差异

回复：不要可能不一样，只是实现的设计架构不一样看不出具体的区别，要深入去阅读边界填充、旋转插值、裁剪精度，深入分析代码运行有没有区别，做出负责认真的分析判断

15. APF地图也调整为旧环境float64吧

16. **坐标取整，应该用round()吧，要不然只要是小数就给int去掉了，位置应该是四舍五入**

```
round(agent_y)  # 旧版本
int(agent.y)    # 新版本可能用int
```

17. 渲染的时候，换成enlarge结果，但是这些信息应该都是实时变化的，感觉缓存的价值不大，因为是实时除草的任务

缓存enlarge结果

if not hasattr(self, '_enlarged_weed_cache'):
self._enlarged_weed_cache = enlarge_map_features(weed_map)