from ..results import Results
from ..measurement import AC_TYPES
import pandas as pd

class TotalEnergyResults(Results):
    """
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date
        `end` is end date
        `active` is (optional) energy in kWh
        `reactive` is (optional) energy in kVARh
        `apparent` is (optional) energy in kVAh
    """
    
    name = "total_energy"

    def append(self, timeframe, new_results):
        """Append a single result.
        e.g. append(TimeFrame(start, end), {'apparent': 34, 'active': 43})
        """
        if set(new_results.keys()) - set(AC_TYPES):
            raise KeyError('new_results must be a combination of ' +
                           str(AC_TYPES))
        super(TotalEnergyResults, self).append(timeframe, new_results)

    def unify(self, other):
        super(TotalEnergyResults, self).unify(other)
        ac_types = set(self._data.columns) - set(['end'])
        for i, row in self._data.iterrows():
            for ac_type in ac_types:
                self._data[ac_type].loc[i] += other._data[ac_type].loc[i]

    def to_dict(self):
        return {'total_energy': self.to_dict()} #.combined().to_dict()}

    def finalize(self):
        self._data = self._data["active"].sum() #self._columns_with_end_removed()].sum() #results_obj._data.sum()['active
        #return self.combined()

    def export_to_cache(self):
        # Can only store in table when Series.
        return pd.Series(self._data) #.fillna(0).convert_objects()

    def import_from_cache(self, cached_stat, sections):
        self._data = cached_stat.loc[0]