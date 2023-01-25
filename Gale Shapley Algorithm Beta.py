import pandas as pd
import numpy as np
import os
from collections import namedtuple
from collections import deque
import json
from collections import defaultdict

os.chdir('/Users/frankylau/Desktop/Projects/Own Projects/Match Making')

json_file = 'User Data/20230109.json'
previous_event_match = 'User Data/previous_match.json'

def transform_json(json_file):
    with open(json_file) as readfile:
        data = json.load(readfile)

    male_self_eval_dict = {}
    male_preference_dict = {}
    female_self_eval_dict = {}
    female_preference_dict = {}

    # Split male and female
    for i in data:
        if int(i['gender']) == 1:
            female_self_eval_dict[i['user_id']] = [int(i['ownstyle']), int(i['ownstyle2']), int(i['ownstyle3']), int(i['ownstyle4']), int(i['age'])]
            female_preference_dict[i['user_id']] = [int(i['targetstyle']), int(i['targetstyle2']), int(i['targetstyle3']), int(i['targetstyle4']), int(i['age'])]
        elif int(i['gender']) == 2:
            male_self_eval_dict[i['user_id']] = [int(i['ownstyle']), int(i['ownstyle2']), int(i['ownstyle3']), int(i['ownstyle4']), int(i['age'])]
            male_preference_dict[i['user_id']] = [int(i['targetstyle']), int(i['targetstyle2']), int(i['targetstyle3']), int(i['targetstyle4']), int(i['age'])]

    return female_self_eval_dict, female_preference_dict, male_self_eval_dict, male_preference_dict

# Determine Ranking Based On Preference
def calculate_euclidean_distance(preference_dict, self_eval_dict):
    output_dict = {}
    temp_dict = {}

    for key, value in preference_dict.items():
        for key_opposite, value_opposite in self_eval_dict.items():
            euclidean_distance = np.linalg.norm(np.array(value) - np.array(value_opposite))

            # Penalize if age difference is more than 5 years
            if abs(int(value[4]) - int(value_opposite[4])) > 5:
                euclidean_distance = euclidean_distance + 4.0

            temp_dict[int(key_opposite)] = euclidean_distance

        output_dict[int(key)] = [k for k, _ in sorted(temp_dict.items(), key=lambda item: item[1])]

    for key, value in output_dict.items():
        copy_value = value.copy()
        copy_value = [i + 10000 for i in copy_value]
        
        # append copy_value to the end of value
        output_dict[key] = value + copy_value

    return output_dict

def pref_to_rank(pref):
    return {a: {b: idx for idx, b in enumerate(a_pref)} for a, a_pref in pref.items()}

def match_making(male_ranking, female_ranking):
    male_list = list(male_ranking.keys())
    female_list = list(female_ranking.keys())
    male_propose_list = {}
    female_propose_list = {}
    
    while len(male_list) > 0:
        male = male_list[0]
        female = list(male_ranking[male].keys())[0]
        
        if female not in female_propose_list:
            female_propose_list[female] = male
            male_propose_list[male] = female
            male_list.remove(male)
        else:
            if female_ranking[female][male] < female_ranking[female][female_propose_list[female]]:
                male_list.remove(male)
                male_list.append(female_propose_list[female])
                male_propose_list.pop(female_propose_list[female])
                female_propose_list[female] = male
                male_propose_list[male] = female

            else:
                male_ranking[male].pop(female)
                male_list.remove(male)
                male_list.append(male)

    df_output = pd.DataFrame(female_propose_list.items(), columns=['A', 'B'])
    df_output['A'] = df_output['A'] % 10000
    
    return df_output, female_propose_list

def populate_data(dict_self_eval, dict_preference):
    copy_self_eval = dict_self_eval.copy()
    copy_preference = dict_preference.copy()

    new_key_self_eval = [int(i) + 10000 for i in copy_self_eval.keys()]
    new_self_eval = dict(zip(new_key_self_eval, list(copy_self_eval.values())))

    new_key_preference = [int(i) + 10000 for i in copy_preference.keys()]
    new_preference = dict(zip(new_key_preference, list(copy_preference.values())))

    dict_self_eval.update(new_self_eval)
    dict_preference.update(new_preference)

    return dict_self_eval, dict_preference

def remove_previous_match(ranking, previous_match, duplicate_json = False):

    for key, value in ranking.items():
        temp_val = int(key) % 10000
        if temp_val in previous_match:
            ranking[key] = [i for i in value if i not in previous_match[temp_val]]

    return ranking

def load_previous_match(json_path):
    temp_dict = defaultdict(list)

    # Load Previous Event Data
    with open(json_path) as readfile:
        previous_match = json.load(readfile)

    for data in previous_match:
        # Append to the list
        temp_dict[int(data['user_id'])].append(int(data['match_user_id']))
    return temp_dict

if __name__ == "__main__":
    # Load Previous Event Data
    previous_match = load_previous_match(previous_event_match)

    # Load Current Event Data
    female_self_eval_dict, female_preference_dict, male_self_eval_dict, male_preference_dict = transform_json(json_file)

    if len(male_preference_dict) < len(female_preference_dict):
        
        female_ranking = calculate_euclidean_distance(female_preference_dict, male_self_eval_dict)
        female_ranking = remove_previous_match(female_ranking, previous_match)
        female_ranking = pref_to_rank(female_ranking)

        male_self_eval_dict, male_preference_dict = populate_data(male_self_eval_dict, male_preference_dict)
        male_ranking = calculate_euclidean_distance(male_preference_dict, female_self_eval_dict)
        male_ranking = remove_previous_match(male_ranking, previous_match)
        male_ranking = pref_to_rank(male_ranking)

        df_output, dict_female_propose = match_making(female_ranking, male_ranking)

        # Flip the output
        df_output_copy = df_output.copy()
        df_output_copy = df_output[['B', 'A']].rename(columns={'B': 'A', 'A': 'B'})

        # Append the flipped output to the original output
        df_output = pd.concat([df_output, df_output_copy], ignore_index=True)

        df_output.to_csv('Output/output.csv', index=False)

    elif len(male_preference_dict) > len(female_preference_dict):

        female_ranking = calculate_euclidean_distance(female_preference_dict, male_self_eval_dict)
        female_ranking = remove_previous_match(female_ranking, previous_match)
        female_ranking = pref_to_rank(female_ranking)

        female_self_eval_dict, female_preference_dict = populate_data(female_self_eval_dict, female_preference_dict)
        male_ranking = calculate_euclidean_distance(male_preference_dict, female_self_eval_dict)
        male_ranking = remove_previous_match(male_ranking, previous_match)
        male_ranking = pref_to_rank(male_ranking)

        df_output, dict_female_propose = match_making(male_ranking, female_ranking)

        # Flip the output
        df_output_copy = df_output.copy()
        df_output_copy = df_output[['B', 'A']].rename(columns={'B': 'A', 'A': 'B'})

        # Append the flipped output to the original output
        df_output = pd.concat([df_output, df_output_copy], ignore_index=True)

        df_output.to_csv('Output/output.csv', index=False)
        
