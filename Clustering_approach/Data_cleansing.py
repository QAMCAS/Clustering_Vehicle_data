import traceback
import arff
from weka.core import jvm
from weka.core.dataset import Instances, Attribute, Instance
import logging
from weka.core.converters import Loader
from weka.filters import Filter, MultiFilter

logger = logging.getLogger(__name__)

try:
    jvm.start(system_cp=True, packages=True, max_heap_size="512m")

except Exception as e:
    print(traceback.format_exc())

"""
The available locations are "Gaimersheim", "Munich" and "Ingolstadt"

"""


location = "Gaimersheim"
loader = Loader("weka.core.converters.ArffLoader")
data_original = loader.load_file('arff_data/' + location + '_Selected_att_Clustering_Weka_Inputdata.arff')

# changing the brake pressure values which are < 0.2 to 0
print(data_original.attribute(2))

for ins in range(0, data_original.num_instances):
    if data_original.get_instance(ins).get_value(2) <= 0.2:
        data_original.get_instance(ins).set_value(2, 0)

data_cleaned = Instances.copy_instances(data_original, 0, data_original.num_instances)

arff.dump('Clustering_input_data/' + location + '/test_data_cleaned.arff',
          data_cleaned,
          relation="Audi",
          names=['timestamps', 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed'])

# Filtering out  attribute 'Timestamps' before  clustering
filter_TS = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
std = Filter(classname="weka.filters.unsupervised.attribute.Standardize")
multi = MultiFilter()
multi.filters = [filter_TS]
multi.inputformat(data_cleaned)
filtered_multi = multi.filter(data_cleaned)

data = Instances.copy_instances(filtered_multi, 0, filtered_multi.num_instances)
arff.dump(
    'Clustering_input_data/' + location + '/test_data_Filtered.arff',
    data, relation="Audi",
    names=['accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed'])

print("done pre-processing the data ")
