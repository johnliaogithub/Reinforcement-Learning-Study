"""
Basically, the idea of the activation probe is that
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

def teacher_process(task_queue, feedback_queue):
    """
    Alice: Generates tasks AND predicts their difficulty using an Activation Probe (PROPEL).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Teacher] Booting on {device}...")
    
    # 1. The Generator (Creates the task)
    generator = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    # 2. The PROPEL Activation Probe (Predicts the reward)
    probe = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)
    
    gen_opt = optim.Adam(generator.parameters(), lr=0.01)
    probe_opt = optim.Adam(probe.parameters(), lr=0.05)
    
    for step in range(100): # Increased steps to let the probe learn
        gen_opt.zero_grad()
        probe_opt.zero_grad()
        
        noise = torch.randn(1, 10, device=device)
        task = generator(noise)
        
        # ==========================================
        # TODO 1: Predict the reward instantly using the probe.
        # Pass the `task` tensor through the `probe` network.
        # ==========================================
        predicted_reward = probe(task)
        
        # Send task to Bob
        task_queue.put(task.detach().cpu())
        
        # Wait for Bob's actual evaluation
        actual_reward = feedback_queue.get()
        actual_reward_tensor = torch.tensor([actual_reward], device=device, dtype=torch.float32)
        
        # ==========================================
        # TODO 2: Calculate Probe Loss
        # Use Mean Squared Error between the predicted_reward and actual_reward_tensor.
        # ==========================================
        probe_loss = nn.functional.mse_loss(predicted_reward, actual_reward_tensor)
        
        # ==========================================
        # TODO 3: Calculate Generator Loss
        # Use dummy REINFORCE, but subtract the predicted_reward from the actual_reward 
        # to act as an advantage baseline. (Remember to detach the prediction!)
        # ==========================================
        gen_loss = - (actual_reward_tensor - predicted_reward) * task.mean()
        
        # ==========================================
        # TODO 4: Backpropagation 
        # If you call backward on probe_loss first, it will destroy the graph 
        # and gen_loss.backward() will crash. Use `retain_graph=True` on the first backward call.
        # ==========================================
        # Call backward on probe_loss here
        probe_loss.backward(retain_graph=True)
        # Call backward on gen_loss here
        gen_loss.backward()
        
        # Guard clause to prevent crashing before TODOs are filled
        if probe_loss is not None and gen_loss is not None:
            gen_opt.step()
            probe_opt.step()
            
            if step % 5 == 0:
                print(f"Step {step:02d} | Actual: {actual_reward:5.2f} | Pred: {predicted_reward.item():5.2f} | Probe Loss: {probe_loss.item():5.2f}")

    task_queue.put("DONE")
    print("[Teacher] Finished training. Shutting down.")


def student_process(task_queue, feedback_queue):
    """
    Bob: Pulls tasks, evaluates them, returns a reward.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.ones(1, 10, device=device)
    
    while True:
        task = task_queue.get()
        if isinstance(task, str) and task == "DONE":
            break
            
        task = task.to(device)
        time.sleep(0.05) # Simulated latency
        
        distance = torch.norm(task - target)
        reward = -torch.abs(distance - 5.0).item() 
        feedback_queue.put(reward)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    feedback_queue = mp.Queue()
    
    p_teacher = mp.Process(target=teacher_process, args=(task_queue, feedback_queue))
    p_student = mp.Process(target=student_process, args=(task_queue, feedback_queue))
    
    p_teacher.start()
    p_student.start()
    
    p_teacher.join()
    p_student.join()
    
    print("[Main] PROPEL simulation complete.")