import os

import yaml

import bddl
import json
import openai
import time
from tqdm import tqdm
from prompt.prompt_utils_og import *
import random

# Set your own API
os.environ['OPENAI_API_KEY'] = "xxx"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())


def main(random_selection=False, headless=False, short_exec=False):
    
    avail_task = ['sorting_mail', 'cleaning_oven', 'unpacking_suitcase', 'storing_food', 'rearranging_furniture', 'polishing_shoes', 'putting_away_Christmas_decorations', 'packing_picnics', 'putting_away_Halloween_decorations', 'laying_wood_floors', 'cleaning_out_drawers', 'setting_up_candles', 'cleaning_up_the_kitchen_only', 'watering_houseplants', 'cleaning_windows', 'waxing_cars_or_other_vehicles', 'bringing_in_wood', 'preserving_food', 'mopping_floors', 're-shelving_library_books', 'cleaning_up_refrigerator', 'chopping_vegetables', 'organizing_boxes_in_garage', 'putting_up_Christmas_decorations_inside', 'filling_an_Easter_basket', 'thawing_frozen_food', 'cleaning_freezer', 'picking_up_take-out_food', 'storing_the_groceries', 'cleaning_barbecue_grill', 'bottling_fruit', 'packing_child_s_bag', 'cleaning_garage', 'boxing_books_up_for_storage', 'packing_boxes_for_household_move_or_trip', 'cleaning_microwave_oven', 'polishing_furniture', 'cleaning_bedroom', 'cleaning_high_chair', 'picking_up_trash', 'cleaning_bathtub', 'locking_every_door', 'cleaning_sneakers', 'installing_a_fax_machine', 'installing_a_scanner', 'cleaning_a_car', 'opening_presents', 'cleaning_toilet', 'packing_lunches', 'cleaning_kitchen_cupboard', 'packing_adult_s_bags', 'cleaning_table_after_clearing', 'moving_boxes_to_storage', 'preparing_a_shower_for_child', 'serving_a_meal', 'filling_a_Christmas_stocking', 'collecting_aluminum_cans', 'putting_leftovers_away', 'assembling_gift_baskets', 'cleaning_cupboards', 'cleaning_stove', 'packing_car_for_trip', 'preparing_salad', 'cleaning_floors', 'putting_dishes_away_after_cleaning', 'packing_bags_or_suitcase', 'making_tea', 'cleaning_up_after_a_meal', 'washing_dishes', 'setting_mousetraps', 'installing_a_modem', 'throwing_away_leftovers', 'sorting_groceries', 'clearing_the_table_after_dinner', 'putting_away_toys', 'defrosting_freezer', 'cleaning_carpets', 'installing_a_printer', 'cleaning_closet', 'brushing_lint_off_clothing', 'organizing_school_stuff', 'cleaning_shoes', 'washing_cars_or_other_vehicles', 'polishing_silver', 'washing_pots_and_pans', 'serving_hors_d_oeuvres', 'collect_misplaced_items', 'vacuuming_floors', 'cleaning_the_pool', 'organizing_file_cabinet', 'locking_every_window', 'installing_alarms', 'loading_the_dishwasher', 'washing_floor', 'cleaning_the_hot_tub', 'opening_packages', 'sorting_books', 'laying_tile_floors', 'packing_food_for_work', 'cleaning_bathrooms']
    print(f"available tasks number: {len(avail_task)}")  # 100

    # random sample 10 tasks from avail_task
    # avail_task = random.sample(avail_task, 10)
    avail_task = avail_task[:5]

    for task in tqdm(avail_task):
        index = avail_task.index(task)
        save_path = os.path.join(os.getcwd(), "respond_og", str(index) + ".json")

        # load task bddl as text
        with open(os.path.join(os.path.dirname(bddl.__file__), "activity_definitions", task, "problem0.bddl")) as f:
            bddl_text = f.read()

        input_context = "Bddl: " + bddl_text + "\n" + "Task: " + task

        messages = get_prompt()

        # Add fewshot_samples
        # samples = get_fewshot_sample()
        # for sample_one in samples:
        #     messages.append({"role": "user", "content": sample_one["context"]})
        #     messages.append({"role": "assistant", "content": sample_one["response"]})
        messages.append({"role": "user", "content": input_context})

        print("================================================")
        print(f"messages: {messages}")

        time.sleep(6)
        while True:
            try:
                response = openai.ChatCompletion.create(
                    # model="gpt-3.5-turbo-0301",
                    model="gpt-3.5-turbo-0613",
                    messages=messages,
                    # temperature=0.5,  # 0.0 - 2.0
                    # max_tokens=2048,
                )
                break
            except:
                time.sleep(20)

        with open(save_path, 'w') as f:
            json_str = json.dumps(response, indent=2)
            f.write(json_str)
            f.write('\n')


if __name__ == "__main__":
    main()
