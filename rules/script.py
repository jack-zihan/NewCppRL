import subprocess
import random

# 定义主程序文件的路径
root = "/home/lzh/NewCppRL/"
# main_script = "rules/jump_path.py"
# dqn_script = "rl/dqn/dqn_test.py"
sac_script = "sac_cont_test.py"


def run_all(seed, difficulty, map_id, obstacle_range, weed_num):
    for weed_dist in ['gaussian', 'uniform']:
        # for noise_set in [[0,0,0], [0.01,1,0.05], [0.02, 2, 0.1], [0.05, 5, 0.2]]:
        for noise_set in [[0,0,0]]:
            with open(root+"rules/env_make.py", 'r+') as file:
                config_content = file.readlines()
                file.seek(0)
                for line in config_content:
                    if "env = gym.make(" in line:
                        line = f"    env = gym.make(id=\"Pasture-v3\", render_mode='rgb_array' if render else None, action_type=\"continuous\", state_size=(128, 128), state_downsize=(128, 128), num_obstacles_range={(obstacle_range[0], obstacle_range[1])}, use_sgcnn=True, use_global_obs=True, use_apf=True, use_box_boundary=True, use_traj=False, noise_position={noise_set[0]}, noise_direction={noise_set[1]}, noise_weed={noise_set[2]})\n"
                    if "obs, info = env.reset(" in line:
                        line = f"    obs, info = env.reset(seed={seed}, options={{'weed_dist': '{weed_dist}', 'map_id': {map_id}, 'weed_num': {weed_num}}})\n"
                    file.write(line)
                file.truncate()

            # # for idx, rl_model in enumerate(["/Users/chuyuliu/CppRL-main-chuyu/ckpt/our_method_new_model.pt", "/Users/chuyuliu/CppRL-main-chuyu/ckpt/baseline_model.pt","/Users/chuyuliu/CppRL-main-chuyu/ckpt/dqn_model_3_0907.pt"]):    
            # for idx, rl_model in enumerate(["/Users/chuyuliu/CppRL-main-chuyu/ckpt/dqn_model_3_0907.pt"]):    
            #     with open("/Users/chuyuliu/CppRL-main-chuyu/rl/dqn/dqn_test.py", "r+") as file:
            #         config_content = file.readlines()
            #         file.seek(0)
            #         for line in config_content:
            #             if "difficulty = " in line:
            #                 line = f"difficulty = \"{difficulty}\"\n"
            #             elif "rl_model =" in line:
            #                 # line = f"rl_model = \"dqn_model_{idx}\"\n"
            #                 line = f"rl_model = \"dqn_model_2\"\n"
            #             elif "weed_dist = " in line:
            #                 line = f"weed_dist = \"{weed_dist}\"\n"
            #             elif "random_seed = " in line:
            #                 line = f"random_seed = {seed}\n"
            #             elif "map_id =" in line:
            #                 line = f"map_id = {map_id}\n"
            #             elif "pt_path =" in line:
            #                 line = f"pt_path = '{rl_model}'\n"
            #             elif "noise_set =" in line:
            #                 line = f"noise_set = {noise_set}\n"
            #             file.write(line)
            #         file.truncate()
                
            
            
                # print("running dqn...")
                # subprocess.run(["python", dqn_script])
                
                files = [
                    root+"/ckpt/sac_cont/2024-09-09_01-16-14_tanhnorm_loc/sac_baseline_continuous_t[01100]_r[2570.25=2509.63~2623.36] - 副本.pt",
                ]
                for file_add in files:
                    before_t = file_add.split('/')[-1]
                    before_t = before_t.split('_t[')[0]
                    with open(root+"rules/sac_cont_test.py", "r+") as file:
                        config_content = file.readlines()
                        file.seek(0)
                        for line in config_content:
                            if "difficulty = " in line:
                                line = f"difficulty = \"{difficulty}\"\n"
                            elif "rl_model =" in line:
                                line = f"rl_model = \"{before_t}\"\n"
                            elif "weed_dist = " in line:
                                line = f"weed_dist = \"{weed_dist}\"\n"
                            elif "random_seed = " in line:
                                line = f"random_seed = {seed}\n"
                            elif "map_id =" in line:
                                line = f"map_id = {map_id}\n"
                            elif "pt_path =" in line:
                                line = f"pt_path = '{file_add}'\n"
                            elif "noise_set =" in line:
                                line = f"noise_set = {noise_set}\n"
                            elif "            action = " in line:
                                line = "            action = logits[2][0].tolist()\n"
                            file.write(line)
                        file.truncate()


                
                    print("running sac...")
                    subprocess.run(["python", sac_script])
                
                
                
            # for idx, rl_model in enumerate(["/Users/chuyuliu/CppRL-main-chuyu/ckpt/sac_baseline_continuous_t[01100]_r[2570.25=2509.63~2623.36] - 副本.pt"]):    
            #     with open("/Users/chuyuliu/CppRL-main-chuyu/rl/sac/sac_cont/sac_cont_test.py", "r+") as file:
            #         config_content = file.readlines()
            #         file.seek(0)
            #         for line in config_content:
            #             if "difficulty = " in line:
            #                 line = f"difficulty = \"{difficulty}\"\n"
            #             elif "rl_model =" in line:
            #                 # line = f"rl_model = \"sac_model_{idx}\"\n"
            #                 line = f"rl_model = \"sac_baseline_continuous\"\n"
            #             elif "weed_dist = " in line:
            #                 line = f"weed_dist = \"{weed_dist}\"\n"
            #             elif "random_seed = " in line:
            #                 line = f"random_seed = {seed}\n"
            #             elif "map_id =" in line:
            #                 line = f"map_id = {map_id}\n"
            #             elif "pt_path =" in line:
            #                 line = f"pt_path = '{rl_model}'\n"
            #             elif "noise_set =" in line:
            #                 line = f"noise_set = {noise_set}\n"
            #             elif "            action = " in line:
            #                 line = "            action = logits[1].argmax().item()\n"
            #             file.write(line)
            #         file.truncate()
            #     with open("/Users/chuyuliu/CppRL-main-chuyu/env_make.py", 'r+') as file:
            #         config_content = file.readlines()
            #         file.seek(0)
            #         for line in config_content:
            #             if "env = gym.make(" in line:
            #                 line = f"    env = gym.make(id=\"Pasture-v3\", render_mode='rgb_array' if render else None, action_type=\"continuous\", state_size=(128, 128), state_downsize=(128, 128), num_obstacles_range={(obstacle_range[0], obstacle_range[1])}, use_sgcnn=True, use_global_obs=True, use_apf=True, use_box_boundary=True, use_traj=False, noise_position={noise_set[0]}, noise_direction={noise_set[1]}, noise_weed={noise_set[2]})\n"
            #             file.write(line)
            #         file.truncate()
                    
            #     print("running sac...")
            #     subprocess.run(["python", sac_script])
                
                
                
                
                
                
                
            # for idx, rl_model in enumerate(["/Users/chuyuliu/CppRL-main-chuyu/ckpt/sac_our_model_1t[01450]_r[2177.22=2158.77~2208.08].pt"]):    
            #     with open("/Users/chuyuliu/CppRL-main-chuyu/rl/sac/sac_test.py", "r+") as file:
            #         config_content = file.readlines()
            #         file.seek(0)
            #         for line in config_content:
            #             if "difficulty = " in line:
            #                 line = f"difficulty = \"{difficulty}\"\n"
            #             elif "rl_model =" in line:
            #                 # line = f"rl_model = \"sac_model_{idx}\"\n"
            #                 line = f"rl_model = \"sac_our_model1\"\n"
            #             elif "weed_dist = " in line:
            #                 line = f"weed_dist = \"{weed_dist}\"\n"
            #             elif "random_seed = " in line:
            #                 line = f"random_seed = {seed}\n"
            #             elif "map_id =" in line:
            #                 line = f"map_id = {map_id}\n"
            #             elif "pt_path =" in line:
            #                 line = f"pt_path = '{rl_model}'\n"
            #             elif "noise_set =" in line:
            #                 line = f"noise_set = {noise_set}\n"
            #             elif "            action = " in line:
            #                 line = "            action = logits[1].argmax().item()\n"
            #             file.write(line)
            #         file.truncate()
            #     with open("/Users/chuyuliu/CppRL-main-chuyu/env_make.py", 'r+') as file:
            #         config_content = file.readlines()
            #         file.seek(0)
            #         for line in config_content:
            #             if "env = gym.make(" in line:
            #                 line = f"    env = gym.make(id=\"Pasture-v2\", render_mode='rgb_array' if render else None, action_type=\"discrete\", state_size=(128, 128), state_downsize=(128, 128), num_obstacles_range={(obstacle_range[0], obstacle_range[1])}, use_sgcnn=True, use_global_obs=True, use_apf=True, use_box_boundary=True, use_traj=False, noise_position={noise_set[0]}, noise_direction={noise_set[1]}, noise_weed={noise_set[2]})\n"
            #             file.write(line)
            #         file.truncate()
            
            
            #     print("running sac...")
            #     subprocess.run(["python", sac_script])
                
                
            
            
                # print("running sac...")
                # subprocess.run(["python", sac_script])

        # for task in ["R_SNAKE", "BCP", "JUMP", "SNAKE", "REACT"]:
        #     with open('/Users/chuyuliu/CppRL-main-chuyu/rules/jump_path.py', 'r+') as file:
        #         config_content = file.readlines()
        #         file.seek(0)
        #         for line in config_content:
        #             if "difficulty =" in line:
        #                 line = f"difficulty = \"{difficulty}\"\n"
        #             elif "weed_dist =" in line:
        #                 line = f"weed_dist = \"{weed_dist}\"\n"
        #             elif "random_seed =" in line:
        #                 line = f"random_seed = {seed}\n"
        #             elif "map_id =" in line:
        #                 line = f"map_id = {map_id}\n"
        #             elif "task_type =" in line and "if" not in line:
        #                 line = f"task_type = \"{task}\"\n"
        #             file.write(line)
        #         file.truncate()

        #     print("running baseline...")
        #     subprocess.run(["python", main_script])



    
for seed in [58, 50 ,72]:
    for hard_degree in ["easy", "medium", "hard"]:
        if hard_degree == "easy":
            for map_id in [2, 3, 6, 16, 20]:
                obstacle_range = [0,0]
                weed_num = 50
                run_all(seed, hard_degree, map_id, obstacle_range, weed_num)
                
                
                
        elif hard_degree == "medium":
            for map_id in [4, 9, 21, 59, 80]:
                obstacle_range = [2,4]
                weed_num = 100
                run_all(seed, hard_degree, map_id, obstacle_range, weed_num)
                
        else:
            for map_id in [22, 29, 57, 63, 22]:
                obstacle_range = [5,8]
                weed_num = 200
                run_all(seed, hard_degree, map_id, obstacle_range, weed_num)
                
                
                

                
                
            
#     with open("/Users/chuyuliu/CppRL-main-chuyu/env_make.py", 'r+') as file:
#             config_content = file.readlines()
#             file.seek(0)
#             for line in config_content:
#                 if "obs, _ = env.reset(" in line:
#                     line = f"obs, info = env.reset(seed={seed}, options={'weed_dist': 'weed_dist':{weed_dist}, 'map_id': {map_id}, 'weed_num': {weed_num}})"

# seed_range = range(0, 30)
# for seed in seed_range:
#     with open("/Users/chuyuliu/CppRL-main-chuyu/env_make.py", 'r+') as file:
#         config_content = file.readlines()
#         file.seek(0)
#         for line in config_content:
#             if "obs, _ = env.reset(" in line:
#                 line = f"obs, info = env.reset(seed={}, options={'weed_dist': 'gaussian', 'map_id': 2, 'weed_num': 50})"

#     with open('/Users/chuyuliu/CppRL-main/rl/dqn/dqn_test.py', 'r+') as file:
#         config_content = file.readlines()
#         file.seek(0)
#         for line in config_content:
#             if "obs, _ = env.reset(seed=" in line:
#                 line = f"obs, _ = env.reset(seed={seed})\n"
#             file.write(line)
#         file.truncate()
#         print(f"Running task: dqn")
#         subprocess.run(["python", main_script])
#     for task in task_types:
#         with open('/Users/chuyuliu/CppRL-main/rules/config.py', 'r+') as file:
#             config_content = file.readlines()
#             file.seek(0)
#             for line in config_content:
#                 if "TASK_TYPE =" in line:
#                     line = f"    TASK_TYPE = \"{task}\"\n"
#                 elif "SEED =" in line:
#                     line = f"    SEED = {seed}\n"
#                 file.write(line)
#             file.truncate()
            
