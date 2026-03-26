import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from examples.inference import GRMInference

model = GRMInference("tanhuajie2001/Robo-Dopamine-GRM-3B")

TASK_INSTRUCTION = "organize the table"
BASE_DEMO_PATH = "./examples/demo_table"
GOAL_IMAGE_PATH = "./examples/demo_table/goal_image.png" 
OUTPUT_ROOT = "./results"

# python /workspace/WMPO/reward_model/rewards/test_rewards.py

output_dir = model.run_pipeline(
    cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    # cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
    # cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
    cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    out_root       = OUTPUT_ROOT,
    task           = TASK_INSTRUCTION,
    frame_interval = 30,
    batch_size     = 1,
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = "incremental",
    visualize      = True
)

print(f"Episode ({BASE_DEMO_PATH}) processed with Incremental-Mode. Output at: {output_dir}")
# 在/nfs/rczhang/code/WMPO/reward_model/rewards/examples/inference.py中修改GPU使用阈值
