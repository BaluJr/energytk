'''
Has to be written as a unit test
'''


def test_slp()
    '''
    Small function to test whether the energy budget 
    works for the slp forecaster.
    '''
    tst = SlpForecaster()
    timesteps = pd.date_range("1.1.2017", freq="15Min", end="2017-12-31 23:45:00")
    overall_power = sum(tst.predict(1000, timesteps))
    avg_power = overall_power / len(timesteps) 
    energy = avg_power * 365 * 24 / 1000 # We want KWH not WH
    assert energy == 1000


