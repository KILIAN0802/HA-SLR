import sys
sys.path.insert(0, '')
import torch
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from Code.Network.SL_GCN.parser import get_parser
from Code.Network.SL_GCN.model.utils import import_class

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    
    with open(p.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Update parser arguments based on the YAML file
    key = list(default_arg.keys())
    for k in default_arg.keys():
        if k not in p.__dict__ or p.__dict__[k] is None:
            p.__dict__[k] = default_arg[k]

    # Function to run evaluation and get prediction scores
    def get_scores(model_name, feeder_name, weights_path, feeder_args):
        Model = import_class(model_name)
        Feeder = import_class(feeder_name)
        
        model = Model(**p.joint_model_args).cuda() # Note: model_args are assumed to be similar
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        
        data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(**feeder_args),
            batch_size=p.batch_size,
            shuffle=False,
            num_workers=p.num_worker)
            
        results = []
        for data, label, index in tqdm(data_loader):
            data = data.float().cuda()
            with torch.no_grad():
                output = model(data)
            results.append(output.data.cpu().numpy())
        
        return np.concatenate(results)

    # Get scores for each model
    r1_scores = get_scores(p.joint_model, p.joint_feeder, p.joint_weights, p.joint_test_feeder_args)
    r2_scores = get_scores(p.bone_model, p.bone_feeder, p.bone_weights, p.bone_test_feeder_args)
    r3_scores = get_scores(p.joint_motion_model, p.joint_motion_feeder, p.joint_motion_weights, p.joint_motion_test_feeder_args)
    r4_scores = get_scores(p.bone_motion_model, p.bone_motion_feeder, p.bone_motion_weights, p.bone_motion_test_feeder_args)

    # Load labels
    with open(p.joint_test_feeder_args['label_path'], 'rb') as f:
        label_info = pickle.load(f)
    
    # Assuming label_info is a list of [name, label] pairs
    label = np.array([item[1] for item in label_info])
    names = [item[0] for item in label_info]

    alpha = [1.0, 0.9, 0.5, 0.5] # Ensemble weights

    right_num = total_num = right_num_5 = 0
    preds = []
    scores = []

    with open('predictions_wo_val_final_test.csv', 'w') as f:
        for i in tqdm(range(len(label))):
            l = label[i]
            name = names[i]
            
            r11 = r1_scores[i]
            r22 = r2_scores[i]
            r33 = r3_scores[i]
            r44 = r4_scores[i]

            score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
            
            rank_5 = score.argsort()[-5:]
            right_num_5 += int(l in rank_5)

            pred = np.argmax(score)
            scores.append(score)
            preds.append(pred)
            right_num += int(pred == l)

            total_num += 1
            f.write('{}, {}\n'.format(name, pred))

        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        print(f'Total samples: {total_num}')
        print(f'Top-1 Accuracy: {acc:.4f}')
        print(f'Top-5 Accuracy: {acc5:.4f}')

    with open('./gcn_ensembled_final_test.pkl', 'wb') as f:
        score_dict = dict(zip(names, scores))
        pickle.dump(score_dict, f)