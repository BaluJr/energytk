from .totalenergy import TotalEnergy
from .goodsections import GoodSections
from .dropoutrate import DropoutRate
from .nonzerosections import NonZeroSections
from .accelerators_stat import get_good_sections_fast

'''
This name space contains the nodes which calculate an statistic/status. 
This is why there are also the Result objects for each node, which collect 
all the chunks and create a final result.
'''