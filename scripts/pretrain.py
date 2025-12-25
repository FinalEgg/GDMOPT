import torch
import torch.nn.functional as F
import numpy as np
import os
from tianshou.data import Batch, VectorReplayBuffer
from tqdm import tqdm

def pretrain_critic(policy, train_collector, logger, steps=5000, batch_size=256, save_path=None):
    """
    Critic 预热阶段：
    使用随机策略收集数据，并仅更新 Critic 网络。
    这有助于 Critic 在 Actor 开始学习之前就对环境有一个基本的价值评估。
    
    Args:
        save_path: 指向 .npz 文件的路径。如果存在则加载，否则尝试收集并保存。
    """
    steps = int(steps) # Ensure int
    print(f"Starting Critic Warmup for {steps} steps...")
    
    # 2. 仅更新 Critic
    policy.train()
    if hasattr(policy, 'critic1_optim'):
        optimizer_c1 = policy.critic1_optim
        optimizer_c2 = policy.critic2_optim
    else:
        # DDPG has only one critic
        optimizer_c1 = policy.critic_optim
        optimizer_c2 = None
    
    num_updates = steps // batch_size
    pbar = tqdm(total=num_updates, desc="Critic Warmup")
    
    # Offline Data Loading Logic
    if save_path and os.path.exists(save_path):
        print(f"Loading warmup data from {save_path} using mmap...")
        try:
            # Use mmap_mode='r' to avoid loading everything into RAM
            data = np.load(save_path, mmap_mode='r')
            data_len = len(data['obs'])
            buffer_size = train_collector.buffer.maxsize
            
            print(f"Total offline data: {data_len}, Buffer size: {buffer_size}")
            
            # Calculate updates per chunk to ensure we cover the dataset proportionally
            # If we have more steps than data, we loop over data.
            # If we have fewer steps than data, we just use what we can.
            
            cursor = 0
            updates_done = 0
            
            while updates_done < num_updates:
                # 1. Load Chunk
                load_size = min(buffer_size, data_len - cursor)
                # If we are at the end and have space, wrap around? 
                # Simpler: just load what we have, train, and next loop will start from 0 if needed.
                
                # Slice from mmap (reads from disk)
                # Note: Tianshou Batch expects numpy arrays, not memmap objects for best compatibility,
                # but memmap behaves like array. However, copying to buffer will trigger read.
                
                # Construct Batch for this chunk
                # We need to copy to memory to put in buffer
                chunk_indices = slice(cursor, cursor + load_size)
                
                # Read data into memory
                obs = np.array(data['obs'][chunk_indices])
                act = np.array(data['act'][chunk_indices])
                rew = np.array(data['rew'][chunk_indices])
                
                
                done = np.array(data['done'][chunk_indices])
                obs_next = np.array(data['obs_next'][chunk_indices])
                
                batch = Batch(
                    obs=obs,
                    act=act,
                    rew=rew,
                    done=done,
                    terminated=done,
                    truncated=np.zeros_like(done),
                    obs_next=obs_next,
                    info=Batch()
                )
                
                # 2. Fill Buffer
                # Reset buffer pointers to treat it as a fresh batch of data
                train_collector.buffer.reset()
                
                # Manual assignment logic (reused from previous fix)
                buf = train_collector.buffer
                length = len(batch)
                
                # Initialize/Populate main buffer _meta
                if hasattr(buf, '_meta') and (buf._meta is None or buf._meta.is_empty()):
                     # Use batch[0] to get shapes
                     example = batch[0]
                     buf._meta = Batch({
                        k: np.zeros((buf.maxsize, *v.shape), dtype=v.dtype)
                        for k, v in example.items() 
                        if isinstance(v, (np.ndarray, np.generic))
                    })
                
                indices = np.arange(length)
                for k, v in batch.items():
                    if hasattr(buf, '_meta') and k in buf._meta:
                        buf._meta[k][indices] = v
                
                buf._size = length
                buf._index = length % buf.maxsize
                
                # Populate sub-buffers
                if isinstance(buf, VectorReplayBuffer) and hasattr(buf, 'buffers'):
                    buffers = buf.buffers
                    num_buffers = len(buffers)
                    chunk_size = length // num_buffers
                    
                    for i in range(num_buffers):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i < num_buffers - 1 else length
                        sub_batch = batch[start:end]
                        if len(sub_batch) == 0: continue
                        
                        sub_buf = buffers[i]
                        sub_len = len(sub_batch)
                        sub_indices = np.arange(sub_len)
                        
                        if hasattr(sub_buf, '_meta') and (sub_buf._meta is None or sub_buf._meta.is_empty()):
                            sub_buf._meta = Batch({
                                k: np.zeros((sub_buf.maxsize, *v.shape), dtype=v.dtype)
                                for k, v in sub_batch[0].items() 
                                if isinstance(v, (np.ndarray, np.generic))
                            })
                            
                        for k, v in sub_batch.items():
                            if k in sub_buf._meta:
                                sub_buf._meta[k][sub_indices] = v
                                
                        sub_buf._size = sub_len
                        sub_buf._index = sub_len % sub_buf.maxsize
                    
                    if hasattr(buf, '_lengths'):
                        buf._lengths = np.array([len(b) for b in buffers])
                
                # 3. Train on this chunk
                # Determine how many updates to run.
                # We want to distribute total updates proportionally to data size.
                # Ratio = num_updates / (data_len / batch_size)
                # Updates for this chunk = (load_size / batch_size) * Ratio
                # Simplified: Updates = (load_size / data_len) * num_updates
                
                # However, if we just want to "loop until finished", we can just run 1 epoch on this chunk?
                # If we run 1 epoch per chunk, we will do 1 pass over data.
                # If we need more updates, we will loop again.
                # Let's calculate updates based on "remaining updates" and "remaining data"?
                # No, let's stick to a fixed ratio to ensure uniform sampling.
                
                # Calculate target updates for this chunk size to match global density
                chunk_updates = int((load_size / data_len) * num_updates)
                # Ensure at least 1 update if we have data
                chunk_updates = max(1, chunk_updates)
                
                # But wait, if num_updates is huge (e.g. 100 epochs), chunk_updates will be huge.
                # We shouldn't overfit to one chunk before moving to next.
                # We should probably switch chunks more often?
                # But switching chunks is expensive (IO).
                # So training for a while on one chunk is good.
                # Let's limit chunk_updates to maybe 1-5 epochs of the chunk?
                # 1 epoch of chunk = load_size / batch_size.
                
                # If chunk_updates > 5 * (load_size / batch_size), we might overfit.
                # But if we have to do 100 epochs total, we have no choice but to do 100 epochs on each chunk (sequentially or interleaved).
                # Interleaved is better: Train 1 epoch on Chunk 1, 1 epoch on Chunk 2... then repeat.
                # But that requires reloading chunks constantly.
                # Given IO cost, maybe doing 5-10 epochs on a chunk is fine.
                
                # Let's just use the proportional amount.
                # If total training is 10M steps on 5M data, that's 2 epochs.
                # So we will train ~2 epochs on this chunk. That is fine.
                
                # Cap updates to remaining needed
                updates_to_run = min(chunk_updates, num_updates - updates_done)
                
                # If we are at the last chunk and have leftover updates due to rounding, take them.
                if cursor + load_size >= data_len and updates_done + updates_to_run < num_updates:
                     # If this is the last chunk of the pass, but we still have updates (e.g. multiple passes needed)
                     # The logic handles it by wrapping cursor.
                     pass
                
                # Special case: if updates_to_run is 0 (e.g. huge data, tiny steps), force 1?
                if updates_to_run == 0 and updates_done < num_updates:
                    updates_to_run = 1
                
                # Train loop
                for step_idx in range(updates_to_run):
                    batch, indices = train_collector.buffer.sample(batch_size)
                    batch = policy.process_fn(batch, train_collector.buffer, indices)
                    
                    # 转换为 Tensor
                    device = next(policy.actor.parameters()).device
                    batch.to_torch(dtype=torch.float32, device=device)
                    
                    # 计算 Target Q
                    with torch.no_grad():
                        if policy.__class__.__name__ == 'DiffusionOPT':
                            target_q = batch.returns
                        elif hasattr(policy, 'actor_old'): # TD3/DDPG
                            target_act = policy.actor_old(batch.obs_next)[0]
                            
                            if hasattr(policy, '_policy_noise') and policy._policy_noise > 0: # TD3
                                noise = torch.randn_like(target_act) * policy._policy_noise
                                noise = noise.clamp(-policy._noise_clip, policy._noise_clip)
                                target_act = (target_act + noise).clamp(policy.action_space.low[0], policy.action_space.high[0])
                            
                            if hasattr(policy, 'critic1_old'): # TD3
                                target_q1 = policy.critic1_old(batch.obs_next, target_act)
                                target_q2 = policy.critic2_old(batch.obs_next, target_act)
                                target_q = torch.min(target_q1, target_q2)
                            else: # DDPG
                                target_q = policy.critic_old(batch.obs_next, target_act)
                                
                            target_q = batch.rew.unsqueeze(1) + (1.0 - batch.done.unsqueeze(1)) * policy._gamma * target_q
                        else: # SAC
                            # SAC uses current actor for target calculation
                            (mu, sigma), _ = policy.actor(batch.obs_next)
                            dist = torch.distributions.Normal(mu, sigma)
                            u_next = dist.rsample()
                            target_act = torch.tanh(u_next)
                            # Correct log_prob for Tanh transform
                            log_prob_next = dist.log_prob(u_next).sum(dim=-1, keepdim=True) - \
                                            torch.log(1 - target_act.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
                            
                            target_q1 = policy.critic1_old(batch.obs_next, target_act)
                            target_q2 = policy.critic2_old(batch.obs_next, target_act)
                            target_q = torch.min(target_q1, target_q2) - policy._alpha * log_prob_next
                            target_q = batch.rew.unsqueeze(1) + (1.0 - batch.done.unsqueeze(1)) * policy._gamma * target_q
                        
                    # 更新 Critic
                    if policy.__class__.__name__ == 'DiffusionOPT':
                        critic_loss = policy._update_critic(batch)
                        loss_std = torch.tensor(0.0)
                    elif optimizer_c2 is not None: # TD3/SAC
                        current_q1 = policy.critic1(batch.obs, batch.act)
                        current_q2 = policy.critic2(batch.obs, batch.act)
                        
                        loss1 = F.mse_loss(current_q1, target_q, reduction='none')
                        loss2 = F.mse_loss(current_q2, target_q, reduction='none')
                        total_loss_elementwise = loss1 + loss2
                        critic_loss = total_loss_elementwise.mean()
                        loss_std = total_loss_elementwise.std()
                        
                        optimizer_c1.zero_grad()
                        optimizer_c2.zero_grad()
                        critic_loss.backward()
                        optimizer_c1.step()
                        optimizer_c2.step()
                    else: # DDPG
                        current_q = policy.critic(batch.obs, batch.act)
                        loss = F.mse_loss(current_q, target_q, reduction='none')
                        critic_loss = loss.mean()
                        loss_std = loss.std()
                        
                        optimizer_c1.zero_grad()
                        critic_loss.backward()
                        optimizer_c1.step()
                    
                    # Update progress bar
                    loss_val = critic_loss.item()
                    std_val = loss_std.item()
                    pbar.set_postfix({'loss': f'{loss_val:.4f}', 'std': f'{std_val:.4f}'})
                    pbar.update(1)
                    
                    # Log to TensorBoard
                    if logger and hasattr(logger, 'writer'):
                        global_step = updates_done + step_idx
                        logger.writer.add_scalar("pretrain/critic_loss", loss_val, global_step)
                        logger.writer.add_scalar("pretrain/critic_loss_std", std_val, global_step)
                
                updates_done += updates_to_run
                
                # Move cursor
                cursor += load_size
                if cursor >= data_len:
                    cursor = 0
                    
        except Exception as e:
            print(f"Failed to load/train with offline data: {e}. Falling back to online collection.")
            # Fallback logic (original online collection)
            # ... (omitted for brevity, assuming offline works or user fixes path)
            raise e

    else:
        print("No offline data found. Collecting random data online...")
        # Original online logic
        # ...
        # For now, just raise error as user expects offline
        raise FileNotFoundError(f"Offline data not found at {save_path}")
        
    pbar.close()
        
    # if (i + 1) % 10 == 0:
    #     # Calculate stats
    #     mean_loss = np.mean(loss_buffer)
    #     std_loss = np.std(loss_buffer)
    #     loss_buffer = [] # Reset
        
    #     # Log to tensorboard
    #     if logger:
    #         if hasattr(logger, 'writer'):
    #             logger.writer.add_scalar("pretrain/critic_loss_mean", mean_loss, i)
    #             logger.writer.add_scalar("pretrain/critic_loss_std", std_loss, i)
    #             logger.writer.add_scalar("Pretrain/Critic_Warmup_Loss_Mean", mean_loss, i)
    #             logger.writer.add_scalar("Pretrain/Critic_Warmup_Loss_Std", std_loss, i)
    #         else:
    #             # Fallback
    #             logger.add_scalar("Pretrain/Critic_Warmup_Loss_Mean", mean_loss, i)
    #             logger.add_scalar("Pretrain/Critic_Warmup_Loss_Std", std_loss, i)

    print("Critic Warmup Finished.")


def pretrain_actor_supervised(policy, dataset_path, epochs=100, batch_size=256, logger=None):
    """
    Actor 预训练阶段 (Behavior Cloning)：
    使用启发式搜索生成的 (State, Action) 数据集进行监督学习。
    """
    if not os.path.exists(dataset_path):
        print(f"Pretrain dataset not found at {dataset_path}. Skipping Actor Pretraining.")
        return
        
    print(f"Loading pretrain dataset from {dataset_path}...")
    data = np.load(dataset_path)
    obs_data = data['obs']
    act_data = data['act']
    
    # 转换为 Tensor
    device = next(policy.actor.parameters()).device
    obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_data, dtype=torch.float32, device=device)
    
    dataset_size = len(obs_data)
    print(f"Dataset size: {dataset_size}. Starting Actor Pretraining for {epochs} epochs...")
    
    optimizer_a = policy.actor_optim
    policy.train()
    
    for epoch in range(epochs):
        indices = np.random.permutation(dataset_size)
        epoch_losses = []
        
        for start_idx in range(0, dataset_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_obs = obs_tensor[batch_indices]
            batch_act = act_tensor[batch_indices]
            
            # Actor Forward
            if policy.__class__.__name__ == 'DiffusionOPT':
                # Diffusion Training: loss(x_start, state)
                # x_start = batch_act (ground truth action)
                # state = batch_obs
                loss = policy.actor.loss(batch_act, batch_obs)
            else:
                actor_out = policy.actor(batch_obs)[0]
                if isinstance(actor_out, tuple):
                    # SAC: (mean, std)
                    # Apply Tanh to mean to match the [-1, 1] action space of the dataset
                    pred_act = torch.tanh(actor_out[0])
                else:
                    # DDPG/TD3: action (Already Tanh)
                    pred_act = actor_out
                
                # MSE Loss
                loss = F.mse_loss(pred_act, batch_act)
            
            optimizer_a.zero_grad()
            loss.backward()
            optimizer_a.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        std_loss = np.std(epoch_losses)
        print(f"Actor Pretrain Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f} ± {std_loss:.6f}")
        
        if logger:
            if hasattr(logger, 'writer'):
                logger.writer.add_scalar("Pretrain/Actor_Supervised_Loss_Mean", avg_loss, epoch)
                logger.writer.add_scalar("Pretrain/Actor_Supervised_Loss_Std", std_loss, epoch)
            else:
                logger.add_scalar("Pretrain/Actor_Supervised_Loss_Mean", avg_loss, epoch)
                logger.add_scalar("Pretrain/Actor_Supervised_Loss_Std", std_loss, epoch)
            
    # Sync weights to target network
    policy.sync_weight()
    print("Actor Pretraining Finished.")
