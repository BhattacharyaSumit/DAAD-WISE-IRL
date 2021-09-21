import torch
import json
from irl_dcb.config import JsonConfig
from irl_dcb.utils import compute_search_cdf, preprocess_fixations
import numpy as np
from irl_dcb.data import LHF_IRL, LHF_Human_Gaze
from irl_dcb.models import LHF_Policy_Cond_Small , LHF_Discriminator_Cond
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import utils
from torch.utils.data import DataLoader
import pickle
import torch.nn.functional as F

hparams = JsonConfig("/content/Scanpath_Prediction/hparams/coco_search18.json")
my_path = "/content/final_valid.txt"
with open(my_path, "rb") as json_file:
  human_scanpaths_valid = pickle.load(json_file)

device = torch.device('cuda:0')

DCB_HR_dir = '/content/Scanpath_Prediction/processed2/DCBs/HR/'
DCB_LR_dir = '/content/Scanpath_Prediction/processed2/DCBs/LR/'
#print(human_scanpaths_valid)
#print('line 22')

if hparams.Train.exclude_wrong_trials:
  human_scanpaths_valid = list(
      filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

trajs_valid = human_scanpaths_valid
#target_annos =  np.load("/content/Scanpath_Prediction/processed/bbox_annos.npy",allow_pickle=True).item()
with open('/content/final_bbox.txt', 'rb') as f:
  target_annos = pickle.load(f)
target_init_fixs = {}
for traj in trajs_valid:
  key = traj['task'] + '_' + traj['name']
  target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)


valid_task_img_pair = np.unique([traj['task'] + '_' + traj['name'] for traj in trajs_valid])
human_mean_cdf, _ = compute_search_cdf(trajs_valid, target_annos,hparams.Data.max_traj_length)
#print('target fixation prob (valid).:', human_mean_cdf)


valid_fix_labels = preprocess_fixations(
        trajs_valid,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length)

cat_names = list(np.unique([x['task'] for x in trajs_valid]))
catIds = dict(zip(cat_names, list(range(len(cat_names)))))

valid_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                     valid_task_img_pair, target_annos, hparams.Data, catIds)

valid_HG_dataset = LHF_Human_Gaze(DCB_HR_dir,DCB_LR_dir,valid_fix_labels,target_annos, hparams.Data,
                    catIds, blur_action=True)


env_valid = IRL_Env4LHF(hparams.Data,max_step=hparams.Data.max_traj_length,
                        mask_size=hparams.Data.IOR_size,
                        status_update_mtd=hparams.Train.stop_criteria,
                        device=device,inhibit_return=True)

#print('line 66')

input_size = 134  # number of belief maps
task_eye = torch.eye(len(catIds)).to(device)

model = LHF_Policy_Cond_Small(hparams.Data.patch_count,len(catIds), task_eye,input_size).to(device)
model2 = LHF_Discriminator_Cond(hparams.Data.patch_count,len(catIds), task_eye,input_size).to(device)
state =  torch.load("/content/Scanpath_Prediction/trained_models/trained_generator.pkg", map_location={'cuda:2': 'cuda:0'})
model.load_state_dict(state["model"])
global_step = state["global_step"]
state2 = torch.load("/content/Scanpath_Prediction/trained_models/trained_discriminator.pkg", map_location={'cuda:2': 'cuda:0'})
model2.load_state_dict(state2["model"])
#optim.load_state_dict(state["optim"])

batch_size = hparams.Train.batch_size
patch_num = hparams.Data.patch_num
max_traj_len = hparams.Data.max_traj_length
im_w = hparams.Data.im_w
im_h = hparams.Data.im_h

valid_img_loader = DataLoader(valid_img_dataset,
                  batch_size= batch_size,shuffle=False, num_workers=16)


#print('line 88')

trajs_all = []
for i_sample in range(1):
  print(i_sample)
  for batch in valid_img_loader:
    env_valid.set_data(batch)
    img_names_batch = batch['img_name']
    cat_names_batch = batch['cat_name']
    with torch.no_grad():
      env_valid.reset() 
      trajs  = utils.collect_trajs(env_valid, model,
                                   patch_num , max_traj_len,
                                   is_eval=False, sample_action=True)
      trajs_all.extend(trajs)
      #all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                           # 'present', trajs['actions'][:, i])
                         # for i in range(env_valid.batch_size)])

#scanpaths = utils.actions2scanpaths(all_actions,patch_num,
                                    #im_w, im_h)

with torch.no_grad():
  for i in range(len(trajs_all)):
    print(i)
    states = trajs_all[i]["curr_states"]
    actions = trajs_all[i]["actions"].unsqueeze(1)
    tids = trajs_all[i]['task_id']
    rewards = F.logsigmoid(model2(states, None, tids))
    trajs_all[i]["rewards"] = rewards


#print(scanpaths)

#utils.cutFixOnTarget(scanpaths, target_annos)
#print(scanpaths)

with open('/content/all_data.txt' , 'wb') as f:
  pickle.dump(trajs_all, f)

#mean_cdf, _ = utils.compute_search_cdf(scanpaths,target_annos,
                                        #max_traj_len)
#print(mean_cdf)
#print(human_mean_cdf)
#print(human_mean_cdf - mean_cdf)
      



