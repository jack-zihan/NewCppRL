"""
GPU Memory Leak Diagnostic Test
Simulates the exact training loop structure to identify memory accumulation points.
"""
import torch
import gc
from tensordict import TensorDict

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_1_baseline_batch_memory():
    """Test 1: Baseline memory for single batch"""
    print("=" * 80)
    print("TEST 1: Baseline Batch Memory")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    initial_mem = get_gpu_memory_mb()

    # Single batch with actual training dimensions
    batch_size = 256
    channels = 5
    resolution = 768

    sampled_td = TensorDict({
        'observation': torch.randn(batch_size, channels, resolution, resolution, device='cuda'),
        'action': torch.randn(batch_size, 2, device='cuda'),
        'reward': torch.randn(batch_size, 1, device='cuda'),
        ('next', 'observation'): torch.randn(batch_size, channels, resolution, resolution, device='cuda'),
        ('next', 'done'): torch.zeros(batch_size, 1, dtype=torch.bool, device='cuda'),
        'index': torch.arange(batch_size, device='cuda')
    }, batch_size=[batch_size])

    after_batch_mem = get_gpu_memory_mb()
    batch_memory = after_batch_mem - initial_mem

    print(f"Single batch memory: {batch_memory:.2f} MB")
    print(f"Expected: ~{2 * batch_size * channels * resolution**2 * 4 / 1024**2:.2f} MB (2 obs × batch × channels × res² × 4 bytes)")

    del sampled_td
    torch.cuda.empty_cache()
    gc.collect()

    return batch_memory

def test_2_losses_tensordict_accumulation():
    """Test 2: Memory retention in losses TensorDict"""
    print("\n" + "=" * 80)
    print("TEST 2: Losses TensorDict Accumulation")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_mem = get_gpu_memory_mb()
    num_updates = 4
    batch_size = 256

    # Simulate the training loop
    losses = TensorDict(batch_size=[num_updates], device='cuda')
    sampled_batches = []

    for i in range(num_updates):
        # Simulate sampling (create large tensors)
        sampled_td = TensorDict({
            'observation': torch.randn(batch_size, 5, 768, 768, device='cuda'),
            'index': torch.arange(batch_size, device='cuda')
        }, batch_size=[batch_size], device='cuda')
        sampled_batches.append(sampled_td)

        # Simulate loss computation with .detach()
        loss_td = TensorDict({
            'loss_actor': torch.randn(batch_size, device='cuda'),
            'loss_qvalue': torch.randn(batch_size, device='cuda'),
            'loss_alpha': torch.randn(batch_size, device='cuda'),
            'td_error': torch.randn(batch_size, device='cuda')
        }, batch_size=[batch_size], device='cuda').detach()

        # Critical line - does this retain sampled_td?
        losses[i] = loss_td.select('loss_actor', 'loss_qvalue', 'loss_alpha')

    after_loop_mem = get_gpu_memory_mb()
    accumulated_memory = after_loop_mem - initial_mem

    print(f"Memory after {num_updates} iterations: {accumulated_memory:.2f} MB")
    print(f"Number of sampled_batches kept in list: {len(sampled_batches)}")

    # Test if losses keeps sampled_batches alive
    del sampled_batches
    gc.collect()
    after_del_batches = get_gpu_memory_mb()
    freed_by_del_batches = after_loop_mem - after_del_batches

    print(f"Memory freed by deleting sampled_batches: {freed_by_del_batches:.2f} MB")

    # Calculate losses.mean() like in training loop
    losses_mean = losses.mean()
    after_mean_mem = get_gpu_memory_mb()

    print(f"Memory after losses.mean(): {after_mean_mem - initial_mem:.2f} MB")

    # Cleanup
    del losses, losses_mean
    torch.cuda.empty_cache()
    gc.collect()

    return accumulated_memory

def test_3_priority_update_reference():
    """Test 3: Does update_tensordict_priority keep references?"""
    print("\n" + "=" * 80)
    print("TEST 3: Priority Update Reference Retention")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_mem = get_gpu_memory_mb()
    batch_size = 256

    # Simulate sampled tensordict with priority info
    sampled_td = TensorDict({
        'observation': torch.randn(batch_size, 5, 768, 768, device='cuda'),
        'index': torch.arange(batch_size, device='cuda'),
        'bucket_id': torch.zeros(batch_size, dtype=torch.long, device='cuda'),
        '_weight': torch.ones(batch_size, device='cuda')
    }, batch_size=[batch_size], device='cuda')

    after_sample_mem = get_gpu_memory_mb()

    # Simulate setting td_error (what happens in update function)
    td_error = torch.randn(batch_size, device='cuda')
    sampled_td.set('td_error', td_error)

    # Simulate what replay buffer does - extract index and td_error
    indices = sampled_td['index']
    errors = sampled_td['td_error']
    priority_data = {'indices': indices, 'errors': errors}

    after_priority_mem = get_gpu_memory_mb()

    print(f"Memory after sampling: {after_sample_mem - initial_mem:.2f} MB")
    print(f"Memory after priority extraction: {after_priority_mem - initial_mem:.2f} MB")

    # Test if priority_data keeps sampled_td alive
    import sys
    print(f"Reference count of sampled_td: {sys.getrefcount(sampled_td)}")

    del sampled_td
    gc.collect()
    after_del_sample = get_gpu_memory_mb()

    print(f"Memory after deleting sampled_td (priority_data still alive): {after_del_sample - initial_mem:.2f} MB")
    print(f"Memory freed: {after_priority_mem - after_del_sample:.2f} MB")

    # Cleanup
    del priority_data, indices, errors, td_error
    torch.cuda.empty_cache()
    gc.collect()

def test_4_gradient_computation_graph():
    """Test 4: Does gradient computation graph retain observation tensors?"""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient Computation Graph Retention")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_mem = get_gpu_memory_mb()
    batch_size = 256
    num_updates = 4

    # Simulate model
    model = torch.nn.Sequential(
        torch.nn.Linear(5 * 768 * 768, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters())
    accumulated_losses = []

    for i in range(num_updates):
        # Simulate batch
        obs = torch.randn(batch_size, 5, 768, 768, device='cuda', requires_grad=False)
        obs_flat = obs.reshape(batch_size, -1)

        # Forward pass
        output = model(obs_flat)
        loss = output.mean()

        # Gradient accumulation
        (loss / num_updates).backward()
        accumulated_losses.append(loss.item())

        # Check memory after each backward
        after_backward_mem = get_gpu_memory_mb()
        print(f"  Iteration {i}: Memory = {after_backward_mem - initial_mem:.2f} MB")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    after_step_mem = get_gpu_memory_mb()
    print(f"Memory after optimizer step: {after_step_mem - initial_mem:.2f} MB")

    # Cleanup
    del model, optimizer, accumulated_losses
    torch.cuda.empty_cache()
    gc.collect()

def test_5_full_training_loop_simulation():
    """Test 5: Full training loop simulation with all components"""
    print("\n" + "=" * 80)
    print("TEST 5: Full Training Loop Simulation")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_mem = get_gpu_memory_mb()
    batch_size = 256
    num_updates = 4

    # Simulate loss module
    class DummyLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5 * 768 * 768, 256).cuda()

        def forward(self, td):
            obs = td['observation'].reshape(td.batch_size[0], -1)
            out = self.linear(obs)
            loss_td = TensorDict({
                'total_loss': out.mean(),
                'loss_actor': out[:, 0].mean(),
                'loss_qvalue': out[:, 1].mean(),
                'loss_alpha': torch.zeros_like(out[:, 0].mean()),
                'alpha': torch.ones_like(out[:, 0].mean()),
                'entropy': torch.zeros_like(out[:, 0].mean()),
                'td_error': out.mean(dim=1)
            }, batch_size=[])
            return loss_td

    loss_module = DummyLoss()
    optimizer = torch.optim.Adam(loss_module.parameters())

    # Training loop
    losses = TensorDict(batch_size=[num_updates], device='cuda')

    for i in range(num_updates):
        # Sample
        sampled_td = TensorDict({
            'observation': torch.randn(batch_size, 5, 768, 768, device='cuda'),
            'index': torch.arange(batch_size, device='cuda'),
            'bucket_id': torch.zeros(batch_size, dtype=torch.long, device='cuda')
        }, batch_size=[batch_size], device='cuda')

        # Loss computation
        loss_out = loss_module(sampled_td)

        # Backward
        (loss_out['total_loss'] / num_updates).backward()

        # Store losses
        losses[i] = loss_out.detach().select('loss_actor', 'loss_qvalue', 'loss_alpha')

        # Priority update simulation (keeps reference to sampled_td)
        sampled_td.set('td_error', loss_out['td_error'].detach())

        iter_mem = get_gpu_memory_mb()
        print(f"  Iteration {i}: Memory = {iter_mem - initial_mem:.2f} MB")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Logging (uses losses and last loss_out)
    losses_mean = losses.mean()
    final_mem = get_gpu_memory_mb()

    print(f"\nFinal memory after loop: {final_mem - initial_mem:.2f} MB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # Cleanup
    del loss_module, optimizer, losses, losses_mean, loss_out
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory tests")
        exit(0)

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # Run all tests
    batch_mem = test_1_baseline_batch_memory()
    accum_mem = test_2_losses_tensordict_accumulation()
    test_3_priority_update_reference()
    test_4_gradient_computation_graph()
    test_5_full_training_loop_simulation()

    # Summary
    print("\n" + "=" * 80)
    print("MEMORY LEAK ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Expected memory per batch: ~{batch_mem:.2f} MB")
    print(f"Observed memory for 4 iterations: {accum_mem:.2f} MB")
    print(f"Expected for 4 batches (linear): ~{4 * batch_mem:.2f} MB")
    print(f"Excess memory: {accum_mem - 4 * batch_mem:.2f} MB")
    print(f"Memory multiplier: {accum_mem / batch_mem:.2f}x (should be ~1.0x with proper cleanup)")
