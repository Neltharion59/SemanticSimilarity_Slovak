from model_management.sts_method_value_persistor import get_persisted_method_values
from model_management.sts_method_pool import sts_method_pool

value_dict = get_persisted_method_values("dataset_sick_all_sk.txt")
for method in value_dict:
    print("{}: {} samples".format(method, len(value_dict[method])))