import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

def teacher_process(task_queue, feedback_queue, num_tasks_to_generate):
    """
    Alice: Generates tasks using a neural network on the GPU. (basically just vectors)
    Learns via a basic RL loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Teacher] Booting on {device}...")
    
    # Tiny Task Generator Network
    generator = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = optim.Adam(generator.parameters(), lr=0.05)
    
    for step in range(10):
        optimizer.zero_grad()
        
        # 1. Generate Task
        noise = torch.randn(1, 10, device=device)
        task = generator(noise) # Keep graph attached!
        
        # We must detach and move to CPU to safely pass across multiprocessing queues
        # Passing attached CUDA tensors over mp.Queue causes memory leaks and NCCL deadlocks
        task_payload = task.detach().cpu()
        
        print(f"[Teacher] Step {step} | Generated Task. Waiting for Bob...")
        
        # ==========================================
        # TODO 1: Put the task_payload into the task_queue
        # ==========================================
        task_queue.put(task_payload)
        
        # ==========================================
        # TODO 2: Block and wait for reward from feedback_queue.
        # ==========================================
        reward = feedback_queue.get()
        
        # 3. RL Update (Dummy REINFORCE: maximize reward via task magnitude)
        # We tie the scalar reward back into the computational graph
        loss = -reward * task.mean() 
        loss.backward()
        optimizer.step()
        
        print(f"[Teacher] Step {step} | Reward: {reward:.3f} | Loss: {loss.item():.3f}\n")

    # Send a poison pill to cleanly shut down the student
    # ==========================================
    # TODO 3: Put the string "DONE" into the task_queue
    # ==========================================
    task_queue.put("DONE")
    
    print("[Teacher] Finished training. Shutting down.")


def student_process(task_queue, feedback_queue):
    """
    Bob: Pulls tasks, evaluates them on the GPU, returns a reward.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Student] Booting on {device}...")
    
    # Bob's secret target
    target = torch.ones(1, 10, device=device)
    
    while True:
        # ==========================================
        # TODO 4: Pull a task from the task_queue.
        # ==========================================
        task = task_queue.get()
        
        if isinstance(task, str) and task == "DONE":
            print("[Student] Received shutdown signal.")
            break
            
        # Move the task back to the GPU for Bob's evaluation
        task = task.to(device)
        
        # Simulate computation time (like running SWE-bench or a compiler)
        time.sleep(0.5)
        
        # "Goldilocks" Objective: The distance should be exactly 5.0. 
        # Too close = too easy. Too far = impossible.
        distance = torch.norm(task - target)
        reward = -torch.abs(distance - 5.0).item() 
        
        # ==========================================
        # TODO 5: Send the reward back to the Teacher via feedback_queue
        # ==========================================
        feedback_queue.put(reward)

if __name__ == "__main__":
    # REQUIRED: PyTorch multiprocessing requires the 'spawn' context to share CUDA tensors safely.
    mp.set_start_method('spawn')
    
    num_tasks_to_generate = 100
    
    # ==========================================
    # TODO 5: Initialize two mp.Queue objects.
    # One for tasks going Teacher -> Student.
    # One for rewards going Student -> Teacher.
    # ==========================================
    task_queue = mp.Queue()   # this is teacher to student
    reward_queue = mp.Queue() # this is student to teacher

    # ==========================================
    # TODO 6: Instantiate and start the two mp.Process objects.
    # Process 1 runs teacher_worker. Process 2 runs student_worker.
    # ==========================================
    teacher = mp.Process(target=teacher_process, args=(task_queue, reward_queue, num_tasks_to_generate))
    student = mp.Process(target=student_process, args=(task_queue, reward_queue))

    teacher.start()
    student.start()
    
    # ==========================================
    # TODO 7: Join the processes to ensure the main script waits for them to finish.
    # ==========================================
    # prevent main script from ending before training finishes
    teacher.join()
    student.join()
    
    print("[Main] All asynchronous self-play workers completed successfully.")